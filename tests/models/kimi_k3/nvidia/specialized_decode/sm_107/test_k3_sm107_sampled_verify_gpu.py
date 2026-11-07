# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_sampled_verify_gpu.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3 specialized decode: the vocabulary-sharded sampled verification
(``runtime.sampled_verify.SampledVerify``) against the stock rejection sampler.

GPU (torchrun, one process per GPU; Rubin, compute capability 10.7, multicast
memory), TP = world: each rank holds its LM-head shard's logits rows [rows,
vocab_rank] BF16 and the full-vocabulary draft-logits cache of which only its own
columns are valid (the other columns are poisoned, as the drafter writes only this
rank's columns). The stock reference (``gather_draft_sampled``,
``rejection_sample``, ``get_num_sampled_and_rejected``: what
``RejectionSampler.__call__`` runs for greedy / temperature-1 requests with no
logits processing, BF16 logits) runs on the full vocabulary on every rank.

Cases: K in {3, 7} x requests {1, 8, 64, 130, 600} (130 x 4 rows and 600 x 4 / 8
rows exceed the exchange's 512-row launch, so its rounds run in several launches) x
standard (probabilistic) / synthetic (conditional rates) / block verification x
probabilistic (draft logits) / one-hot drafts. Requests mix greedy and temperature 1,
random 64-bit and 31-bit seeds, permuted request slots, -1 placeholder drafts (a step
that starts within the prefill) and chunked-prefilling requests (num_sampled 0).
Checked per request, bitwise: num_sampled, num_rejected and the output tokens up to
the stock sampled count (accepted drafts and the recovered / bonus token); every rank
the same result; a second call (the exchanges' parity hand-off) the same result.

    torchrun --nproc-per-node 4 \
        tests/models/kimi_k3/nvidia/specialized_decode/sm_107/test_k3_sm107_sampled_verify_gpu.py \
        [standard] [synthetic] [block]
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("triton")
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

VOCAB = 163840  # Kimi K3 / K3.1 vocabulary: 40960 per rank at TP4 (2048-aligned)
WORLD = 4
STEPS = (3, 7)
BATCHES = (1, 8, 64, 130, 600)
MODES = ("standard", "synthetic", "block")
PAD = 3  # input tokens ahead of the batch's logits rows (logits_indices offset)
POISON = 30.0  # off-shard draft-logits columns and logits row padding
MAX_LISTED = 8  # mismatching requests listed per case


def _rubin_world():
    """At least WORLD Rubin GPUs (compute capability 10.7: the gating of
    ``VerifyArgmax.build``, which ``SampledVerify`` is built behind)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return all(
        torch.cuda.get_device_capability(i) == (10, 7) for i in range(WORLD)
    )


@pytest.mark.parametrize("part", MODES)
def test_gpu_torchrun(part):
    """The torchrun checks below at TP4 (skipped without 4 Rubin GPUs)."""
    import subprocess

    if not _rubin_world():
        pytest.skip(f"needs {WORLD}+ Rubin GPUs (compute capability 10.7, multicast)")
    command = [sys.executable, "-m", "torch.distributed.run"]
    command += [f"--nproc-per-node={WORLD}", __file__, part]
    subprocess.run(command, check=True, timeout=3600)


# ------------------------------------------------------------------------- inputs


def _case(steps, num_reqs, world, rank, device, group):
    """A decode batch of ``num_reqs`` requests with ``steps`` drafts each, identical
    on every rank (small tensors from a CPU generator, the large ones broadcast from
    rank 0), as the runner's InputBatch / sampler state fields."""
    import torch.distributed as dist

    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

    q = steps + 1
    rows = num_reqs * q
    max_reqs = num_reqs + 5
    cpu = torch.Generator().manual_seed(1000 * steps + num_reqs)
    gen = torch.Generator(device=device).manual_seed(7 + 1000 * steps + num_reqs)
    req = torch.arange(num_reqs)

    # Request slots (permuted, some unused), sampling state per slot.
    slots = torch.randperm(max_reqs, generator=cpu)[:num_reqs]
    greedy = (req + steps) % 3 == 0  # B1: greedy at K3, sampled at K7
    temperature = torch.where(torch.rand(max_reqs, generator=cpu) < 0.5, 0.0, 1.0)
    temperature[slots] = torch.where(greedy, 0.0, 1.0)
    seeds = torch.randint(0, 2**62, (max_reqs,), generator=cpu, dtype=torch.int64)
    small = slots[req % 4 == 3]
    seeds[small] = seeds[small] % 2**31
    base = torch.randint(16, 200_000, (num_reqs,), generator=cpu)
    prefill_len = torch.randint(1, 1 << 20, (max_reqs,), generator=cpu)
    prefill_len[slots] = (base * torch.rand(num_reqs, generator=cpu)).long().clamp(1)
    # Placeholder drafts (the step starts within the prefill: P/D first step) and
    # chunked prefilling (seq_len < prefill_len: nothing sampled).
    placeholder = req % 11 == 5
    chunked = req % 13 == 7
    prefill_len[slots[placeholder]] = base[placeholder] + 1
    prefill_len[slots[chunked]] = base[chunked] + q + 4
    seq_lens = base + q

    # Tokens: input 0 the last sampled token, inputs 1..K the drafts; row j verifies
    # input j + 1. The target logits of row j boost its draft token by 0 (a random
    # token: rejected), 9, 12, 15 (accepted with a probability) or 30 (accepted).
    tokens = torch.randint(0, VOCAB, (num_reqs, q), generator=cpu)
    pick = torch.randint(0, 5, (num_reqs, steps), generator=cpu)
    boosts = torch.tensor([0.0, 9.0, 12.0, 15.0, 30.0])[pick]
    sigma = torch.tensor([0.5, 1.0, 2.0])[req % 3]
    tau = torch.tensor([0.02, 0.3, 1.5])[(req // 3) % 3]
    input_ids = torch.randint(0, VOCAB, (PAD + rows,), generator=cpu)
    input_ids[PAD:] = tokens.flatten()
    positions = torch.randint(0, 1 << 20, (PAD + rows,), generator=cpu)
    positions[PAD:] = (base[:, None] + torch.arange(q)).flatten()

    full = torch.randn(
        (rows, VOCAB), generator=gen, device=device, dtype=torch.bfloat16
    )
    full.view(num_reqs, q, VOCAB).mul_(
        sigma.to(device, torch.bfloat16)[:, None, None]
    )
    boosted = (req[:, None] * q + torch.arange(steps)).flatten().to(device)
    drafted = tokens[:, 1:].flatten().to(device)
    full[boosted, drafted] = (
        full[boosted, drafted].float() + boosts.flatten().to(device)
    ).bfloat16()

    # Draft logits (pre-temperature) by request slot: the target row plus noise.
    cache = DraftLogitsCache(
        max_reqs, steps, VOCAB, dtype=torch.bfloat16, fill=0.0, device=device
    )
    cache.values.normal_(generator=gen)
    for start in range(0, num_reqs, 64):
        end = min(start + 64, num_reqs)
        target = full.view(num_reqs, q, VOCAB)[start:end, :steps].float()
        noise = torch.randn(target.shape, generator=gen, device=device)
        noise.mul_(tau[start:end].to(device)[:, None, None]).add_(target)
        cache.values[slots[start:end].to(device)] = noise.bfloat16()
        del target, noise
    for tensor in (full, cache.values):
        dist.broadcast(tensor, 0, group=group)

    expanded_idx = slots.repeat_interleave(q)
    uncond = [0.9 * 0.8**i for i in range(steps)]

    def dev(tensor, dtype):
        return tensor.to(device=device, dtype=dtype).contiguous()

    case = SimpleNamespace(
        steps=steps,
        num_reqs=num_reqs,
        rows=rows,
        greedy=greedy,
        placeholder=placeholder,
        chunked=chunked,
        slots=slots,
        full=full,
        cache=cache,
        input_ids=dev(input_ids, torch.int32),
        positions=dev(positions, torch.int64),
        logits_indices=dev(PAD + torch.arange(rows), torch.int64),
        expanded_local_pos=dev(torch.arange(q).repeat(num_reqs), torch.int32),
        idx_mapping=dev(slots, torch.int64),
        expanded_idx_mapping=dev(expanded_idx, torch.int64),
        cu_num_logits=dev(torch.arange(0, rows + 1, q), torch.int32),
        seq_lens=dev(seq_lens, torch.int32),
        prefill_len=dev(prefill_len, torch.int32),
        temperature=dev(temperature, torch.float32),
        seeds=dev(seeds, torch.int64),
        rates=torch.tensor(
            unconditional_to_conditional_rates(uncond),
            dtype=torch.float32,
            device=device,
        ),
    )

    # This rank's inputs: the logits shard (a row-padded view at B8 / B130, the
    # padding poisoned; contiguous otherwise) and the draft-logits cache with every
    # other rank's columns poisoned.
    vocab_rank = VOCAB // world
    col0 = rank * vocab_rank
    if num_reqs in (8, 130):
        buffer = torch.full(
            (rows, vocab_rank + 64), POISON, dtype=torch.bfloat16, device=device
        )
        case.shard = buffer[:, :vocab_rank]
        case.shard.copy_(full[:, col0 : col0 + vocab_rank])
    else:
        case.shard = full[:, col0 : col0 + vocab_rank].contiguous()
    case.shard_draft = cache.values.clone()
    case.shard_draft[..., :col0] = POISON
    case.shard_draft[..., col0 + vocab_rank :] = POISON
    return case


# ----------------------------------------------------------------------- the runs


def _stock(c, mode, has_draft):
    """``RejectionSampler.__call__``'s verification on the full vocabulary. Returns
    (sampled [num_reqs, K + 1], raw num_sampled before the chunked-prefill zeroing,
    num_sampled, num_rejected)."""
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
        gather_draft_sampled,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    draft_sampled, pos = gather_draft_sampled(
        c.input_ids,
        c.positions,
        c.logits_indices,
        c.expanded_idx_mapping,
        c.expanded_local_pos,
        c.prefill_len,
    )
    sampled, num_sampled = rejection_sample(
        c.full,
        c.cache if has_draft else None,
        draft_sampled,
        c.cu_num_logits,
        pos,
        c.idx_mapping,
        c.expanded_idx_mapping,
        c.expanded_local_pos,
        c.temperature,
        c.seeds,
        c.steps,
        c.rates if mode == "synthetic" else None,
        use_fp64=False,
        use_block_verification=mode == "block",
    )
    raw = num_sampled.clone()
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled, c.seq_lens, c.cu_num_logits, c.idx_mapping, c.prefill_len
    )
    return sampled, raw, num_sampled, num_rejected


def _sharded(verify, c, mode, has_draft):
    """``KimiLinearForCausalLM.verify_sampled``'s ``verify.run`` call with the
    kwargs ``RejectionSampler.verify_sharded`` passes."""
    return verify.run(
        c.shard,
        c.steps,
        input_ids=c.input_ids,
        positions=c.positions,
        logits_indices=c.logits_indices,
        expanded_local_pos=c.expanded_local_pos,
        idx_mapping=c.idx_mapping,
        seq_lens=c.seq_lens,
        prefill_len=c.prefill_len,
        temperature=c.temperature,
        seeds=c.seeds,
        draft_logits=c.shard_draft if has_draft else None,
        synthetic_rates=c.rates if mode == "synthetic" else None,
        block_verification=mode == "block",
    )


# --------------------------------------------------------------------- comparison


def _packed(tokens, num_sampled, num_rejected, valid):
    """The compared words: tokens where ``valid`` (else -2) and both counts."""
    masked = torch.where(valid, tokens, torch.full_like(tokens, -2))
    return torch.cat(
        [masked.flatten(), num_sampled.long().flatten(), num_rejected.long().flatten()]
    )


def _describe(c, r, stock, shard):
    s_tok, raw, s_ns, s_nr = (t.cpu() for t in stock)
    h_tok, h_ns, h_nr = (t.cpu() for t in shard)
    k, n = c.steps, int(raw[r])
    tags = []
    tags.append("greedy" if bool(c.greedy[r]) else "T=1")
    if bool(c.placeholder[r]):
        tags.append("placeholder drafts")
    if bool(c.chunked[r]):
        tags.append("chunked prefill")
    diff = (s_tok[r, :n] != h_tok[r, :n]).nonzero().flatten().tolist()
    if diff:
        j = diff[0]
        kind = "accepted draft" if j < n - 1 else ("bonus" if j == k else "recovered")
        where = f"first token diff @{j} ({kind}), {len(diff)} position(s)"
    else:
        where = "tokens equal, counts differ"
    return (
        f"    req {r} slot {int(c.slots[r])} [{', '.join(tags)}]: stock "
        f"num_sampled {int(s_ns[r])} (raw {n}) rejected {int(s_nr[r])} | sharded "
        f"num_sampled {int(h_ns[r])} rejected {int(h_nr[r])}; {where}; stock "
        f"{s_tok[r, :n].tolist()} sharded {h_tok[r, : max(n, 1)].tolist()}"
    )


def _check(verify, c, mode, has_draft, world, rank, group):
    import torch.distributed as dist

    label = (
        f"{mode} {'draft-logits' if has_draft else 'one-hot'} K{c.steps} "
        f"B{c.num_reqs} ({c.rows} rows)"
    )
    stock = _stock(c, mode, has_draft)
    s_tok, raw, s_ns, s_nr = stock
    shard = _sharded(verify, c, mode, has_draft)
    again = _sharded(verify, c, mode, has_draft)
    h_tok, h_ns, h_nr = shard
    q = c.steps + 1
    lines = []
    if (
        h_tok.shape != (c.num_reqs, q)
        or h_tok.dtype != torch.int64
        or h_ns.shape != (c.num_reqs,)
        or h_ns.dtype != torch.int32
        or h_nr.dtype != torch.int32
    ):
        lines.append(
            f"  output layout: sampled {tuple(h_tok.shape)} {h_tok.dtype}, "
            f"num_sampled {tuple(h_ns.shape)} {h_ns.dtype}, num_rejected {h_nr.dtype}"
        )
    else:
        valid = torch.arange(q, device=raw.device)[None, :] < raw.long()[:, None]
        token_bad = ((s_tok != h_tok) & valid).any(1)
        bad = token_bad | (s_ns != h_ns) | (s_nr != h_nr)
        bad_reqs = bad.nonzero().flatten().tolist()
        if bad_reqs:
            # Where the differences are: per request its first differing position.
            kinds = {"accepted draft": 0, "recovered": 0, "bonus": 0, "count only": 0}
            s_cpu, h_cpu, raw_cpu = s_tok.cpu(), h_tok.cpu(), raw.cpu()
            for r in bad_reqs:
                n = int(raw_cpu[r])
                d = (s_cpu[r, :n] != h_cpu[r, :n]).nonzero().flatten().tolist()
                if not d:
                    kinds["count only"] += 1
                elif d[0] < n - 1:
                    kinds["accepted draft"] += 1
                else:
                    kinds["bonus" if d[0] == c.steps else "recovered"] += 1
            lines.append(
                f"  {len(bad_reqs)}/{c.num_reqs} requests differ from stock "
                f"(first diff: {kinds}); requests {bad_reqs[:32]}"
                + (" ..." if len(bad_reqs) > 32 else "")
            )
            lines += [_describe(c, r, stock, shard) for r in bad_reqs[:MAX_LISTED]]
        first = _packed(h_tok, h_ns, h_nr, valid)
        second = _packed(*again, valid)
        if not torch.equal(first, second):
            lines.append("  a second call (next exchange parity) differs from the 1st")
        # Every rank computes every decision: the same result on every rank.
        gathered = [torch.empty_like(first) for _ in range(world)]
        dist.all_gather(gathered, first, group=group)
        differ = [p for p in range(world) if not torch.equal(gathered[p], gathered[0])]
        if differ:
            lines.append(f"  ranks {differ} differ from rank 0")

    # Coverage: the case must exercise varied acceptance (a draw-independent
    # outcome, e.g. every draft accepted or none, would hide a wrong uniform).
    live = ~(c.placeholder | c.chunked)
    accepted = (raw.cpu().long() - 1).clamp(min=0)
    sampled_live = live & ~c.greedy
    hist_s = torch.bincount(accepted[sampled_live], minlength=q).tolist()
    hist_g = torch.bincount(accepted[live & c.greedy], minlength=q).tolist()
    if c.num_reqs >= 64:
        partial = [a for a in accepted[sampled_live].tolist() if 0 < a < c.steps]
        if len(set(accepted[sampled_live].tolist())) < 2 or not partial:
            lines.append(f"  coverage: sampled accepted lengths {hist_s} too uniform")

    failed = torch.tensor([1 if lines else 0], dtype=torch.int32, device=s_tok.device)
    dist.all_reduce(failed, group=group)
    if lines:
        print(f"[rank {rank}] FAIL {label}\n" + "\n".join(lines), flush=True)
    elif rank == 0:
        print(
            f"ok   {label}: greedy {int((live & c.greedy).sum())} sampled "
            f"{int(sampled_live.sum())} placeholder {int(c.placeholder.sum())} "
            f"chunked {int(c.chunked.sum())}; accepted-length histogram sampled "
            f"{hist_s} greedy {hist_g}",
            flush=True,
        )
    return int(failed.item() > 0)


# ---------------------------------------------------------------------- torchrun


def _gpu_main(parts):
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    assert VOCAB % world == 0
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.sampled_verify import (
        SampledVerify,
    )

    # As SampledVerify.build: TP group size / rank, the device group, the LM head's
    # per-rank vocabulary (rank r owns columns [r * vocab_rank, (r + 1) * vocab_rank)).
    verify = SampledVerify(world, rank, group, VOCAB // world, device)
    failures = 0
    for steps in STEPS:
        for num_reqs in BATCHES:
            case = _case(steps, num_reqs, world, rank, device, group)
            for mode in parts:
                for has_draft in (True, False):
                    args = verify, case, mode, has_draft, world, rank, group
                    failures += _check(*args)
            del case
            torch.cuda.empty_cache()
    if rank == 0:
        print(f"sampled verify parity: {failures} failing case(s)", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    # torchrun ... test_k3_sm107_sampled_verify_gpu.py [standard]
    # [synthetic] [block] (default: all three)
    sys.exit(1 if _gpu_main(sys.argv[1:] or MODES) else 0)
