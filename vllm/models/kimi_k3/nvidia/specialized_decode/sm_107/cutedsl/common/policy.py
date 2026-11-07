# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch policy tables."""

_NONE = object()


class BatchTable(dict):
    """``batch_table``'s result: a dict of the closed ranges' batches, and the value of
    an open-ended range ("65-:1": every batch from 65 on) for the batches past the
    materialized ones. Lookups (``get``, ``[]``, ``in``) see every batch, and
    ``values`` holds every value the table takes (the materialized batches reach one
    past the largest bound the spec names). An open table has no finite set of
    batches: iteration, ``keys``, ``items``, ``len``, ``dict(table)``, ``{**table}``
    and ``|`` raise TypeError instead of silently dropping the open range
    (``materialized`` is the explicit view); ``copy`` and pickling keep it."""

    def __init__(self, items=(), tail=_NONE, top=0, first=0):
        super().__init__(items)
        # the open range's value (past ``top``: no closed range reaches there) and its
        # first batch
        self.tail, self.top, self.first = tail, top, first

    def _tail(self, batch):
        if self.tail is _NONE or not isinstance(batch, int) or batch <= self.top:
            return _NONE
        return self.tail

    def __missing__(self, batch):
        value = self._tail(batch)
        if value is _NONE:
            raise KeyError(batch)
        return value

    def __contains__(self, batch):
        return super().__contains__(batch) or self._tail(batch) is not _NONE

    def get(self, batch, default=None):
        if super().__contains__(batch):
            return super().__getitem__(batch)
        value = self._tail(batch)
        return default if value is _NONE else value

    @property
    def open(self):
        """Whether an open-ended range covers every batch past ``top``."""
        return self.tail is not _NONE

    def _finite(self, what):
        if self.open:
            raise TypeError(
                f"{what} of an open-ended batch table (every batch from {self.first} "
                "on): it has no finite set of batches; use get / [] / in, values, or "
                "materialized()"
            )

    def __iter__(self):
        self._finite("iteration")
        return super().__iter__()

    def __len__(self):
        self._finite("len")
        return super().__len__()

    def __bool__(self):
        return self.open or super().__len__() > 0

    def keys(self):
        self._finite("keys")
        return super().keys()

    def items(self):
        self._finite("items")
        return super().items()

    def __or__(self, other):
        self._finite("|")
        return super().__or__(other)

    def __ror__(self, other):
        self._finite("|")
        return super().__ror__(other)

    def materialized(self):
        """The materialized batches as a plain dict (an open range's batches past
        ``top`` are not in it)."""
        return dict(super().items())

    def copy(self):
        return BatchTable(super().items(), self.tail, self.top, self.first)

    __copy__ = copy

    def __reduce__(self):  # (pickle, deepcopy: the open range survives)
        tail = self.tail if self.open else None
        return _rebuild, (self.materialized(), self.open, tail, self.top, self.first)

    def update(self, other=(), **kwargs):
        """Later entries override earlier ones: ``other``'s open range (if any) covers
        every batch from its first on, this table's included."""
        if isinstance(other, BatchTable):
            if other.open:
                first, top = other.first, max(self.top, other.top)
                if self.open:  # this table's tail up to the new top, then other's
                    for b in range(self.top + 1, top + 1):
                        super().setdefault(b, self.tail)
                for b in [b for b in super().keys() if b >= first]:
                    del self[b]
                self.tail, self.top, self.first = other.tail, top, first
                super().update({b: other.tail for b in range(first, top + 1)})
            other = other.materialized()
        super().update(other, **kwargs)


def _rebuild(items, is_open, tail, top, first):
    return BatchTable(items, tail if is_open else _NONE, top, first)


def batch_table(spec):
    """Per-batch table of integers ("1-2:96,3-8:120"), tuples ("16:64.16.4.8") or names
    ("16-64:trtllm"); later entries override earlier ones. An open-ended range ("65-:1")
    covers every batch from its first one on (``BatchTable``)."""
    items, tail, first, top = [], _NONE, 0, 0
    for item in filter(None, spec.split(",")):
        span, value = item.split(":")
        lo, dash, hi = span.partition("-")
        if "." in value:
            value = tuple(map(int, value.split(".")))
        elif value.lstrip("-").isdigit():
            value = int(value)
        if dash and not hi:  # open-ended
            items.append((int(lo), None, value))
            tail, first = value, int(lo)
            top = max(top, int(lo))
        else:
            items.append((int(lo), int(hi or lo), value))
            top = max(top, int(hi or lo))
    if tail is not _NONE:
        top += 1  # (one batch past every closed range: the open range's own value)
    table = BatchTable(tail=tail, top=top, first=first)
    for lo, hi, value in items:
        # an open range is materialized up to ``top``
        last = top if hi is None else hi
        dict.update(table, {b: value for b in range(lo, last + 1)})
    return table  # past ``top`` only open ranges apply: the last one's value
