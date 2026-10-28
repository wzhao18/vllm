# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact speedups for Kimi K3 chat tokenization (off by default).

K3's tokenizer (the checkpoint's ``tokenization_kimi.TikTokenTokenizer``)
renders a chat into XTML ``EncodeSegment`` objects and encodes every segment
on its own: ``_encode_chat_segments`` concatenates
``_encode_text_piece(segment.text, segment.allow_special)`` over the segments.
The ids of a segment therefore depend only on its text and its
``allow_special`` flag, which makes two exact speedups possible:

``VLLM_K3_FAST_SPLIT=1``
    ``_encode_text_piece`` runs a pure-Python per-character loop
    (``_split_whitespaces_or_nonwhitespaces``) that cuts runs of whitespace or
    non-whitespace longer than 25k characters (tiktoken issue 195). When no
    such run exists, that loop yields the text whole. The fast path proves
    this with a few C-level regex searches (one per 12.5k-character window)
    and only falls back to the original loop for text that contains a run of
    at least half the limit.

``VLLM_K3_TOKCACHE_BYTES=<bytes>`` (``K``/``M``/``G`` suffixes, binary)
    A token-id cache over runs of segments. A run is one rendered message:
    the segments up to and including the ``<|end_of_msg|>`` control segment
    (the generation prompt is the last run). The key is a 128-bit SHA-256
    digest, salted per process, of the run's exact segment texts and flags;
    the value is the run's ids as an int32 array. Agentic clients re-send the
    whole conversation every turn, so every message but the new ones hits.
    Bookkeeping: an ``OrderedDict`` in LRU order, O(1) per lookup, insert and
    eviction, bounded by the byte cap (ids + key + entry overhead), guarded by
    a lock (the renderer's tokenizer thread pool may have several workers).
    One cache per API-server process.

Both speedups are installed only when the checkpoint's three methods match
the source this module was written against and a self-test reproduces the
original ids.
"""

import hashlib
import inspect
import logging
import os
import sys
import textwrap
import threading
import types
from array import array
from collections import OrderedDict
from typing import Any

import regex as re

logger = logging.getLogger(__name__)

END_OF_MSG = "<|end_of_msg|>"

# sha256 of textwrap.dedent(inspect.getsource(method)) for the checkpoint's
# tokenization_kimi.py (moonshotai/Kimi-K3, tokenization_kimi.py sha256
# f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944).
_PINNED_SOURCES = {
    "_encode_chat_segments": (
        "46ab075f05531cffc00555ac428179378f71a5c9de03aafc06f55e1cfaddb2b2"
    ),
    "_encode_text_piece": (
        "0d77618e639452f3c9c1a810f3d3d072e2a6f47e6ab293f0dd39e6bcfe50e115"
    ),
    "_split_whitespaces_or_nonwhitespaces": (
        "b8f5acb3521037a488896765515d2569bc658ae7129ba1f202d5104565a9ad08"
    ),
}


def parse_bytes(value: str | None) -> int:
    """``"512M"`` -> 536870912; empty / ``"0"`` -> 0."""
    if not value:
        return 0
    v = value.strip().upper().removesuffix("IB").removesuffix("B")
    mult = 1
    if v and v[-1] in "KMGT":
        mult = 1 << (10 * ("KMGT".index(v[-1]) + 1))
        v = v[:-1]
    return int(float(v) * mult)


def _source_digest(fn: Any) -> str | None:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()


def _whitespace_classes() -> tuple[Any, Any]:
    """Regexes for one ``str.isspace`` / one non-``str.isspace`` character,
    built from ``str.isspace`` itself (``regex``'s ``\\s`` omits U+001C-1F)."""
    ws = "".join(f"\\u{c:04x}" for c in range(0x110000) if chr(c).isspace())
    return re.compile(f"[{ws}]"), re.compile(f"[^{ws}]")


def make_fast_split(orig_split):
    """Exact replacement for ``_split_whitespaces_or_nonwhitespaces``.

    The original cuts ``s`` only inside a run (of whitespace, or of
    non-whitespace, as ``str.isspace`` decides) longer than ``max_len``;
    otherwise it yields ``s`` whole. With ``h = (max_len + 2) // 2``, any run
    of ``max_len + 1`` or more characters fully contains a window
    ``[k * h, k * h + h)``. So when every such window holds both a
    whitespace and a non-whitespace character, no cut happens and ``s`` is
    the only slice. Anything else falls back to the original loop, so the
    slices are identical in every case.
    """
    ws_search, nws_search = (p.search for p in _whitespace_classes())

    def fast_split(s: str, max_len: int):
        n = len(s)
        if n > max_len:
            h = (max_len + 2) // 2
            for p in range(0, n - h + 1, h):
                if ws_search(s, p, p + h) is None or nws_search(s, p, p + h) is None:
                    yield from orig_split(s, max_len)
                    return
        yield s

    return fast_split


class K3TokenCache:
    """Byte-bounded LRU of run digest -> int32 token ids."""

    # OrderedDict slot and node per entry beyond sys.getsizeof(key) +
    # sys.getsizeof(value): 127 B measured with tracemalloc (CPython 3.12).
    ENTRY_OVERHEAD = 128

    def __init__(self, cap_bytes: int, max_entry_frac: float = 0.25):
        self.cap_bytes = cap_bytes
        self.max_entry_bytes = int(cap_bytes * max_entry_frac)
        self._d: OrderedDict[bytes, array] = OrderedDict()
        self._lock = threading.Lock()
        self.nbytes = 0
        self._salted = hashlib.sha256(os.urandom(16))
        self._key_overhead = sys.getsizeof(b"\0" * 16) + self.ENTRY_OVERHEAD

    def __len__(self) -> int:
        return len(self._d)

    def key(self, texts: list[str], flags: bytes) -> bytes:
        """Prefix-free digest of the run: segment count, flags, the length of
        every text (code points), then the concatenated texts."""
        h = self._salted.copy()
        h.update(len(flags).to_bytes(8, "little"))
        h.update(flags)
        h.update(array("q", map(len, texts)).tobytes())
        h.update("".join(texts).encode("utf-8", "surrogatepass"))
        return h.digest()[:16]

    def get(self, key: bytes) -> array | None:
        with self._lock:
            ids = self._d.get(key)
            if ids is not None:
                self._d.move_to_end(key)
            return ids

    def put(self, key: bytes, ids: array) -> None:
        size = sys.getsizeof(ids) + self._key_overhead
        if size > self.max_entry_bytes:
            return
        with self._lock:
            d = self._d
            if key in d:
                d.move_to_end(key)
                return
            d[key] = ids
            self.nbytes += size
            while self.nbytes > self.cap_bytes:
                _, old = d.popitem(last=False)
                self.nbytes -= sys.getsizeof(old) + self._key_overhead

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.nbytes = 0


class K3TokenizerAccel:
    """What ``install`` put on a tokenizer."""

    def __init__(self, cache: K3TokenCache | None, fast_split: bool):
        self.cache = cache
        self.fast_split = fast_split


def _encode_runs(
    tok,
    segments,
    cache: K3TokenCache | None,
) -> list[int]:
    encode_piece = tok._encode_text_piece
    out = array("i")
    texts: list[str] = []
    flags = bytearray()
    n = len(segments)
    for i, seg in enumerate(segments):
        text = seg.text
        special = seg.allow_special
        texts.append(text)
        flags.append(1 if special else 0)
        if (special and text == END_OF_MSG) or i == n - 1:
            ids = None
            key = None
            if cache is not None:
                key = cache.key(texts, bytes(flags))
                ids = cache.get(key)
            if ids is None:
                piece_ids: list[int] = []
                for t, f in zip(texts, flags):
                    piece_ids.extend(encode_piece(t, allow_special_tokens=bool(f)))
                ids = array("i", piece_ids)
                if key is not None:  # key is only set with a cache
                    cache.put(key, ids)  # type: ignore[union-attr]
            out.extend(ids)
            texts = []
            flags = bytearray()
    return out.tolist()


def _reference_ids(cls, tok, segments) -> list[int]:
    """The checkpoint's own encoder and splitter, on a proxy that shares only
    the tiktoken model (the pinned sources use nothing else)."""
    ref = types.SimpleNamespace(
        model=tok.model,
        _split_whitespaces_or_nonwhitespaces=cls._split_whitespaces_or_nonwhitespaces,
    )
    ref._encode_text_piece = types.MethodType(cls._encode_text_piece, ref)
    return cls._encode_chat_segments(ref, segments)


def _self_test(tok, cls, segment_cls) -> str | None:
    """Encode probe segments both ways; return a reason on any difference."""
    probes = [
        ("<|open|>", True),
        ("message", False),
        (" role", False),
        ('="', False),
        ("user", False),
        ('"', False),
        ("<|sep|>", True),
        ("def f(x):\n    return x  # <|im_end|> stays text\n" * 50, False),
        ("x" * 30_000 + " " * 26_000 + "y" * 12_600 + " z", False),
        ("中文 テキスト    tabs\t\tand\r\nnewlines", False),
        ("", False),
        ("<|close|>", True),
        ("message", False),
        ("<|sep|>", True),
        (END_OF_MSG, True),
        ("<|open|>", True),
        ("think", False),
        ("<|sep|>", True),
    ]
    segments = [segment_cls(t, allow_special=s) for t, s in probes if t]
    ref = _reference_ids(cls, tok, segments)
    got = tok._encode_chat_segments(segments)
    if got != ref:
        return "self-test ids differ"
    again = tok._encode_chat_segments(segments)  # all hits
    if again != ref:
        return "self-test cached ids differ"
    split = tok.__dict__.get("_split_whitespaces_or_nonwhitespaces")
    if split is not None:
        orig = cls._split_whitespaces_or_nonwhitespaces
        for text, _ in probes:
            for n in (25_000, 7, 1):
                if list(split(text, n)) != list(orig(text, n)):
                    return f"self-test split differs (n={n})"
    return None


def maybe_install(tokenizer) -> K3TokenizerAccel | None:
    """Install the env-selected speedups on a K3 tokenizer instance."""
    fast = os.environ.get("VLLM_K3_FAST_SPLIT", "0") not in ("", "0")
    cap = parse_bytes(os.environ.get("VLLM_K3_TOKCACHE_BYTES"))
    if not fast and cap <= 0:
        return None
    tok = tokenizer
    cls = type(tok)
    needed = (
        "_encode_chat_segments",
        "_encode_text_piece",
        "_split_whitespaces_or_nonwhitespaces",
    )
    if not all(hasattr(cls, m) for m in needed):
        logger.warning(
            "K3 tokenizer speedups requested but %s lacks %s; leaving it as is",
            cls.__name__,
            needed,
        )
        return None
    for m in needed:
        got = _source_digest(getattr(cls, m))
        if got != _PINNED_SOURCES[m]:
            logger.warning(
                "K3 tokenizer speedups requested but %s.%s differs from the "
                "pinned source (sha256 %s); leaving the tokenizer as is",
                cls.__name__,
                m,
                got,
            )
            return None
    module = sys.modules.get(cls._encode_chat_segments.__module__)
    segment_cls = getattr(module, "EncodeSegment", None)
    if segment_cls is None:
        try:
            segment_cls = module.build_chat_segments.__globals__["EncodeSegment"]  # type: ignore[union-attr]
        except (AttributeError, KeyError):
            segment_cls = None
    if segment_cls is None:
        logger.warning("K3 tokenizer speedups: no EncodeSegment class found")
        return None

    cache = K3TokenCache(cap) if cap > 0 else None
    accel = K3TokenizerAccel(cache, fast)
    if fast:
        tok.__dict__["_split_whitespaces_or_nonwhitespaces"] = make_fast_split(
            cls._split_whitespaces_or_nonwhitespaces
        )

    if cache is not None:

        def _encode_chat_segments(self, segments):
            return _encode_runs(self, segments, cache)

        tok.__dict__["_encode_chat_segments"] = types.MethodType(
            _encode_chat_segments, tok
        )
    why = _self_test(tok, cls, segment_cls)
    if cache is not None:
        cache.clear()
    if why is not None:
        tok.__dict__.pop("_encode_chat_segments", None)
        tok.__dict__.pop("_split_whitespaces_or_nonwhitespaces", None)
        logger.warning("K3 tokenizer speedups disabled: %s", why)
        return None
    logger.info(
        "K3 tokenizer speedups on: fast split %s, token cache %s",
        "on" if fast else "off",
        f"{cap} bytes" if cache is not None else "off",
    )
    return accel
