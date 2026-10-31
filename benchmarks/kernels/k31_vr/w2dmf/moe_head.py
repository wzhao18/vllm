# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE head per layer from a rank trace: attn_res end -> MegaMoE start (critical path), and MegaMoE end -> next attn_res start."""
import gzip, json, sys, statistics as st
for f in sys.argv[1:]:
    ev = json.load(gzip.open(f))['traceEvents']
    k = sorted((e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')), key=lambda e: e['ts'])
    heads, tails, moe = [], [], []
    for i, e in enumerate(k):
        if 'mega_moe' not in e['name']:
            continue
        ar = [c for c in k[max(0, i - 30):i] if 'attn_res' in c['name']]
        if ar:
            heads.append(e['ts'] - (ar[-1]['ts'] + ar[-1]['dur']))
        moe.append(e['dur'])
        nx = [c for c in k[i + 1:i + 30] if 'attn_res' in c['name']]
        if nx:
            tails.append(nx[0]['ts'] - (e['ts'] + e['dur']))
    print(f"{f.split('/raw/')[-1]}: MoE layers {len(moe)} head med {st.median(heads):.2f} us | megamoe med {st.median(moe):.1f} | tail (megamoe end -> next attn_res) med {st.median(tails):.2f}")
