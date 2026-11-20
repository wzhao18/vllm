# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA round 6: floors of every per-token sparse-MLA prefill form on VR200, from MEASURED terms only.

One F16K DSA layer per GPU = 16384 tokens x 2048 top-k keys x 96 heads, FP8 KV rows of 576 B
(19.33 GB of (token, key) row deliveries), 2473 64-key units per SM on 212 SMs. Kernel time in-step
= cycles / step clock (the 2.3 kW limiter acts on the 20-50 ms step average, not per kernel).

Measured terms (runs under k31ws/runs/P-MLA unless noted):
  F_STEP   in-step SM clock during the sparse-MLA calls, 1.96-2.18 GHz, mean 2.10 (M28 clkp, step_p r6clk1)
  DELIV    random 576 B row L2->SMEM cap 57.8-60 B/clk/SM for every mechanism and mix (M27 run190;
           P-MLA-2CTA g10a 60), clock-proportional (pmla13 run181)
  TC_QK    QK latent 8 x M128 N64 K64 TS at the full 2x-K rate = 32 clk/MMA (P-MLA-2CTA pmla11, 4 issuers)
  TC_ROPE  RoPE QK as SS ~96 clk per 64 keys (pmla11)
  TC_PV    PV M128 N96 K64 with A = V^T MN-major from SMEM ~92 clk/MMA (103 B/clk operand fetch, pmla11)
  MIX_H8   exact h8 tensor mix 778-790 clk per 64 keys with 2-4 issuing warps (pmla11 m11a-c)
  CHAIN    single-S round trip with the exact QK, rt13a stand-in, compile-time layouts: h8 661,
           L1 (48-key S) 698, L2 (32-key S) 890 clk per 64 keys (M32 run195; P-MLA-2CTA rt13a h8 595)
  KOM_SS   keys-on-M 2-CTA SS 775 clk per 64 keys (P-MLA-2CTA probe3/5); 1-CTA SS ~860 (calibrated)
  H9       2-CTA keys-on-M TS: K and V^T into TMEM through registers, LDG <= 51 B/clk -> >= 723 clk;
           the pair's PV needs the partner keys' V half (832 B/key) -> >= 887 clk (P-MLA-2CTA (3));
           DSMEM 15-21 B/clk/SM (hwdb)
  KERNEL   production kernel in-step 1157 clk per 64 keys (M28), 1342 us/layer (P-MLA-2CTA 8fea934fd8)
TMEM is 576 columns: h8 = Q 128 + S 64 + O^T 384; RoPE-Q in TMEM forces one S of <= 48 columns;
FA4-style O[heads, dims] (512 cols) needs Q from SMEM (+72 KB/64 keys of operand fetch).
"""

UNITS = 2473  # 64-key units per SM per F16K layer
ROW = 576
TARGET_US = 790.0  # 1.1 x model achievable 718 us/layer (int13-p-prof.real.gap.json r5 row)


def us(clk_per_unit: float, ghz: float) -> float:
    return clk_per_unit * UNITS / (ghz * 1e3)


def rows():
    deliv = (64 * ROW / 60.0, 64 * ROW / 57.8)  # clk per unit at the delivery cap
    h8_tc = 8 * 32 + 96 + 4 * 92  # perfect-issue tensor work of the h8 form
    l1_tc = 12 * 24 + 4 * 92  # RoPE TS, N48: 12 QK MMAs per 64 keys at full rate + PV
    return [
        ("delivery cap (any per-token form; not a structure)", deliv[0], deliv[1]),
        ("h8 family TC floor, perfect 4-issuer issue (h8, h10a, qkh, qks, any schedule)", h8_tc, h8_tc),
        ("h8 measured tensor mix (2-4 issuers)", 778, 790),
        ("h8 single-S chain (rt13a 595; M32 probe 661)", 595, 661),
        ("h8 + L1 free-running QK warp + PV warp mix (M32 run195; = pmla11 2-issuer)", 789, 789),
        ("L1 RoPE-TS + one 48-key S: chain (M32; scaled by rt13a/probe 628)", 628, 698),
        ("L1 tensor work at perfect 4-warp QK issue (theory; measured 2-warp mix = h8 789)", l1_tc, l1_tc),
        ("L2 RoPE-TS + one 32-key S: chain (M32)", 890, 890),
        ("FA4-style O[heads,dims] with Q SS (+72 KB/unit at 128 B/clk)", h8_tc + 72 * 1024 / 128, h8_tc + 72 * 1024 / 128),
        ("keys-on-M 2-CTA SS (measured)", 775, 775),
        ("keys-on-M 1-CTA SS (calibrated)", 860, 860),
        ("2-CTA keys-on-M TS (h9): register producer / pair V", 723, 887),
        ("production kernel today (in-step, M28)", 1157, 1157),
    ]


def main():
    print(f"{'form':80s} {'clk/64 keys':>14s} {'us @2.10':>10s} {'us @1.96-2.18':>16s}")
    for name, lo, hi in rows():
        clk = f"{lo:.0f}" if lo == hi else f"{lo:.0f}-{hi:.0f}"
        print(f"{name:80s} {clk:>14s} {us(lo, 2.10):10.0f} {us(lo, 2.18):7.0f}-{us(hi, 1.96):.0f}")
    print(f"target {TARGET_US:.0f} us/layer = {TARGET_US * 2.10e3 / UNITS:.0f} clk per 64 keys at 2.10 GHz")


if __name__ == "__main__":
    main()
