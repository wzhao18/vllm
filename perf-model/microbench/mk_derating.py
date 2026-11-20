#!/usr/bin/env python3
"""Build perf-model/inputs/derating.json from the microbench lane rows (2026-11-18 synthesis)."""
import json, collections, statistics, os
PM = '/Users/weizha/k31vr-dev/perf-model'
R = PM + '/microbench/results'
ld = lambda p: json.load(open(p))
T = ld(R + '/model/targets.json')['targets']
dec = ld(R + '/decode/rows.json')
pre = ld(R + '/prefill/rows.json')
g512 = ld(R + '/gap512/rows.json')
# wave 2 (2026-11-18): D3-32 EP32 decode, same-session D3 control + release NVFP4-MoE arm, P2b / P5-4 / P5-8 prefill and
# release NVFP4 arms. Their t_ach (if any) came from a mid-run optables; every point below also gets the CURRENT model's
# achievable (model_ach_ms_current / derate_current, computed here), which is what calibrate.py uses.
w2 = ld(R + '/wave2/rows.json')
for r in w2:
    if r.get('kind') == 'step':
        tag = r['case'].split(':')[0]
        dec.append(dict(r, case='w2:' + r['case'], moe_quant='nvfp4' if tag.startswith('d3rel') else 'mxfp4_mxfp8'))
    elif r.get('row') == 'full step':
        pre.append(r)

def ach(topo, band, ck='mxfp4_release'):
    return T[f'{ck}|{topo}|{band}']['step_ach_ms']

# ---------------- per-topology: decode step rows (dedupe identical r/k, prefer non-reuse) ----------------
dsteps = collections.defaultdict(list)
for r in dec:
    if r['kind'] != 'step':
        continue
    dsteps[(r['topology'], r['k'], r['r_per_gpu'], r.get('moe_quant', 'mxfp4_mxfp8'))].append(r)
for r in g512:
    if r['group'] == 'STEP' and r['topology'].startswith('D3'):
        B = int(r['case'].split('_B')[1].split('_')[0])
        dsteps[('D3', 3, B / 4, 'mxfp4_mxfp8')].append({'case': 'gap512:' + r['case'], 'measured_us': r['measured_us'],
                                          't_ach_us': r['t_ach_us'], 'derate': r['derate'], 'notes': r['notes']})
topo = {}
for (tp, k, rr, mq), rows in sorted(dsteps.items()):
    meas = statistics.median(x['measured_us'] for x in rows) / 1000
    old = [x['t_ach_us'] for x in rows if x.get('t_ach_us') and not x['case'].startswith('w2:')]
    a = old[0] / 1000 if old else None
    e = topo.setdefault(tp, {'phase': 'decode', 'points': [], 'source_rows': []})
    e['points'].append({'k': k, 'r_per_gpu': rr, 'moe_quant': mq, 'measured_ms': round(meas, 2),
                        'model_ach_ms': round(a, 2) if a else None,
                        'derate': round(meas / a, 3) if a else None, 'n_rows': len(rows),
                        'kernel_debug': bool(a and meas / a > 2.0)})
    e['source_rows'] += [('decode:' + x['case']) if not x['case'].startswith(('gap512', 'w2:')) else x['case'] for x in rows]

def band_of(rr):
    return 'r<=0.25' if rr <= 0.25 else 'r0.5-2' if rr <= 2 else 'r3-8' if rr <= 8 else 'r12-24'

for tp, e in topo.items():
    bands = collections.defaultdict(list)
    for p in e['points']:
        if p['derate'] is None or p.get('moe_quant', 'mxfp4_mxfp8') != 'mxfp4_mxfp8':
            continue
        if tp == 'D1' and p['k'] == 7 and p['r_per_gpu'] == 8:
            continue  # MoE path-switch cliff, excluded from trend (KERNEL-DEBUG listed separately)
        bands[band_of(p['r_per_gpu'])].append(p['derate'])
    e['derate_by_band'] = {b: {'median': round(statistics.median(v), 3), 'min': min(v), 'max': max(v), 'n': len(v)}
                           for b, v in bands.items()}
topo['D1']['notes'] = 'specialized decode TP16=DCP16, MoE-TP16 (no EP); k7 r8 cliff (51.2 ms, MoE path switch at 1024 rows) excluded from band derates'
topo['D2']['notes'] = 'ran MoE-TP8 (sdec refuses EP8 token-shard); model D2 is EP8 -> not like-for-like at r>=4'
topo['D3-32'] = dict(topo['D3-32'], notes='wave 2: DP8 x (TP4=DCP4) SP, EP32, vendored megamoe_rubin (deep_gemm situ); KV 2329 blocks/rank, 13 blocks/request, r_max ~44.8; r32 bench OOM, r48 refused (SP 512 verify-row cap)')
topo['D3']['notes'] = 'vendored megamoe_rubin (older kernel); SP path uses generic cuBLAS split-K BF16 GEMMs; gap512 C512 rows merged (B32-B92 = r8-23)'

# ---------------- per-topology: prefill step rows ----------------
for r in pre:
    if r['row'] != 'full step':
        continue
    tp = r['topology']
    rel = 'release' in r['case']
    e = topo.setdefault(tp, {'phase': 'prefill', 'points': [], 'source_rows': []})
    tokgpu = r['notes'].split(' tok/s/GPU')[0]
    e['points'].append({'case': r['case'], 'moe_quant': 'nvfp4 (release default, not like-for-like)' if rel else 'mxfp4_mxfp8',
                        'measured_ms': round(r['measured_us'] / 1000, 1), 'model_ach_ms': round(r['t_ach_us'] / 1000, 1),
                        'derate': r['derate'], 'measured_tok_s_gpu': float(tokgpu)})
    e['source_rows'].append(('wave2:' if r.get('wave') == 2 else 'prefill:') + r['case'])
for r in g512:
    if r['group'] == 'STEP' and r['topology'].startswith('P2'):
        e = topo['P2']
        e['points'].append({'case': 'gap512 C512 ' + r['case'], 'moe_quant': 'mxfp4_mxfp8',
                            'measured_ms': round(r['measured_us'] / 1000, 1), 'model_ach_ms': round(r['t_ach_us'] / 1000, 1),
                            'derate': r['derate']})
        e['source_rows'].append('gap512:' + r['case'])
for tp in ['P1', 'P2', 'P3-16', 'P4-4', 'P4-2', 'P2b', 'P5-4', 'P5-8']:
    v = [p['derate'] for p in topo[tp]['points'] if p['moe_quant'] == 'mxfp4_mxfp8']
    topo[tp]['derate_step'] = {'median': round(statistics.median(v), 3), 'min': min(v), 'max': max(v), 'n': len(v)}
topo['P1']['notes'] = 'P1 AttnRes model error (code runs AttnRes on 1/8 rows + AG; model prices all rows) offsets ~20 ms of excess'
topo['P3-16']['notes'] = '16K/GPU needs pinned 14 GiB KV + mns 4 (weights 225 GiB at TP1)'
topo['P4-4']['notes'] = topo['P4-2']['notes'] = 'PR #20 tree, no scheduler; owner = batch index mod TP; DSA KV replicated'

# ---------------- projections for unmeasured topologies (analog transfer) ----------------
# decode: ratio transfer (step_ach x analog derate) and additive transfer (step_ach + analog excess ms)
analog_d = {'D5': 'D3', 'D6': 'D3', 'D7-4-16': 'D3', 'D7-2-16': 'D3', 'D7-8-16': 'D3',
            'D4-16': 'D3', 'D9-16': 'D3', 'D8-16': 'D1'}
meas_at = {}
for tp in ['D1', 'D2', 'D3']:
    for p in topo[tp]['points']:
        if p['k'] == 3 and p['model_ach_ms'] and p.get('moe_quant', 'mxfp4_mxfp8') == 'mxfp4_mxfp8':
            meas_at[(tp, p['r_per_gpu'])] = p
def analog_pt(tp, rr):
    if (tp, rr) in meas_at:
        return meas_at[(tp, rr)]
    if tp == 'D3' and rr == 32:
        return meas_at[('D3', 24.0)]  # nearest measured
    return None
proj = {}
for band, rr in [('decode_r1', 1.0), ('decode_r8', 8.0), ('decode_r32', 32.0)]:
    rows = {}
    for tp in ['D1', 'D2', 'D3', 'D4-16', 'D5', 'D6', 'D7-4-16', 'D7-2-16', 'D7-8-16', 'D8-16', 'D9-16']:
        key = f'mxfp4_release|{tp}|{band}'
        if key not in T:
            continue
        a = T[key]['step_ach_ms']
        m = meas_at.get((tp, rr))
        if not m and tp == 'D3' and rr == 32:
            m = meas_at[('D3', 24.0)]
            rows[tp] = {'model_ach_ms': a, 'nearest_measured': 'r24 (B96): 33.34 ms vs model r24 18.84 (1.77); measured KV max r ~33', 'kind': 'measured-nearby'}
            continue
        if m:
            rows[tp] = {'model_ach_ms': a, 'measured_ms': m['measured_ms'], 'derate': m['derate'], 'kind': 'measured'}
            continue
        an = analog_d.get(tp)
        ap = analog_pt(an, rr) if an else None
        if not ap:
            rows[tp] = {'model_ach_ms': a, 'kind': 'unmeasured; over measured KV capacity (D1 k3 r_max ~15.6)' if tp in ('D1','D8-16') else 'unmeasured'}
            continue
        lo = a + (ap['measured_ms'] - ap['model_ach_ms'])
        hi = a * ap['measured_ms'] / ap['model_ach_ms']
        rows[tp] = {'model_ach_ms': a, 'proj_ms_additive': round(lo, 2), 'proj_ms_ratio': round(hi, 2),
                    'analog': f"{an} r{ap['r_per_gpu']} k3 (derate {ap['derate']})", 'kind': 'projected'}
    proj[band] = rows

# prefill: tok/s/GPU = 16384 / N_attn / step; analogs
Nattn = {'P1': 8, 'P2': 4, 'P2b': 8, 'P3-16': 1, 'P4-4': 4, 'P4-2': 2, 'P5-4': 4, 'P5-8': 8, 'P6-16': 2, 'P6-4-16': 4, 'P9': 1, 'P9-16': 1}
analog_p = {'P2b': 'P2', 'P5-4': 'P2', 'P5-8': 'P1', 'P6-16': 'P4-2', 'P6-4-16': 'P4-4', 'P9': 'P3-16', 'P9-16': 'P3-16'}
# dedup is not implemented: the measured MegaMoE vs the dedup-model rows is 1.09-1.14 vs 0.89 without dedup.
DEDUP_MOE_PENALTY = 1.11 / 0.89
pp = {}
for tp, N in Nattn.items():
    key = f'mxfp4_release|{tp}|prefill_F16K_L78K'
    a = T[key]['step_ach_ms']
    model_tps = 16384 / N / (a / 1000)
    if tp in ('P1', 'P2', 'P3-16', 'P4-4', 'P4-2', 'P2b', 'P5-4', 'P5-8'):
        p16 = [p for p in topo[tp]['points'] if p['moe_quant'] == 'mxfp4_mxfp8' and 'L78k' in p['case'] and ('16k' in p['case'] or tp == 'P3-16')]
        p16 = [p for p in p16 if '16k' in p['case']] or p16
        d = p16[0]['derate']
        pp[tp] = {'model_tok_s_gpu': round(model_tps), 'derate': d, 'derated_tok_s_gpu': round(model_tps / d), 'kind': 'measured', 'case': p16[0]['case']}
    else:
        an = analog_p[tp]
        d = topo[an]['derate_step']['median']
        step = a * d
        note = f'analog {an} step derate {d}'
        if tp.startswith('P6'):
            ex = sum(o['count'] * o['t_ach_us'] for o in T[key]['ops'] if o['fam'] == 'experts') / 1000
            step += ex * (DEDUP_MOE_PENALTY - 1)
            note += f'; + no-dedup MoE penalty {ex * (DEDUP_MOE_PENALTY - 1):.1f} ms (dedup not implemented)'
        if tp.startswith('P9'):
            note += '; DSA-CP not implemented (no code path)'
        pp[tp] = {'model_tok_s_gpu': round(model_tps), 'derated_tok_s_gpu': round(16384 / N / (step / 1000)), 'kind': 'projected', 'note': note}
proj['prefill_F16K_L78K'] = pp

# ---------------- family derates (curated from lane rows) ----------------
fam = {
 'decode': {
  'experts (MegaMoE, EP16/EP8 fused dispatch+GEMM+combine)': {
    'derate': 1.30, 'range': [0.86, 1.33], 'path': 'vendored megamoe_rubin in-step (D3, gap512); FI 6e438852b standalone: 1.19-1.32 at 4 tok/rank, 0.86-1.06 at 32-128',
    'derate_fi_6e438852b': {'tok_rank<=4': 1.27, 'tok_rank 32-128': 0.97},
    'source_rows': ['gap512:FAMILY MegaMoE B32-B92 (1.27-1.33)', 'gap512:MOE.megamoe per layer (1.25-1.31)', 'decode:prof3 mega_moe_situ (1.31 r8, 1.56 r1)', 'moe:EP16 MX t4/t32/t128 (1.32/1.06/0.96)', 'moe:EP8 MX t4/t32 (1.25/0.94)']},
  'experts (MoE-TP16 sdec rubin, D1)': {'derate': 1.19, 'range': [1.19, 1.19], 'path': 'kernel only; front/tail +4.8 ms fused comm not included',
    'source_rows': ['decode:prof1b sdec MoE experts kernel (r8)']},
  'BF16 dense GEMMs, SP path (cuBLAS nvjet + CuTe split-K, D3-type)': {'derate': 2.3, 'range': [1.95, 2.65],
    'source_rows': ['gap512:FAMILY BF16 dense GEMMs B32-B92 (1.95-2.51)', 'decode:prof3 nvjet BF16 GEMMs (2.35 r8, 2.65 r1)']},
  'BF16 dense GEMMs, TP path (sdec CuTe, D1-type) and standalone M<=1024': {'derate': 1.5, 'range': [1.26, 1.65],
    'source_rows': ['decode:prof1b o-proj+AR (1.26)', 'ops1:GEMM BF16 M<=64 time-weighted 1.37', 'ops1:GEMM BF16 64<M<=1024 time-weighted 1.65']},
  'MXFP8 GEMMs M<=1024 (nvfp4_modelopt shared/KDA)': {'derate': 2.2, 'range': [1.95, 2.48], 'source_rows': ['ops1:GEMM MXFP8 M<=1024 time-weighted']},
  'KDA core decode (conv+gate+recurrence+o_norm, FP32 state)': {'derate': 2.2, 'range': [1.93, 11.4],
    'note': 'floor-dominated: ~0.6-0.8 ms/step floor + ~2.4x slope; r<=1 derates 9.7-11.4',
    'source_rows': ['gap512:FAMILY KDA core (1.93-2.43)', 'decode:prof3 sdec KdaWide (2.67 r8, 9.71 r1)', 'decode:prof1b KdaVerify (3.44 r8, 11.43 r1)', 'ops1:KDA decode tw 3.7 (bench writes per-token checkpoints)']},
  'DSA q path + sparse MLA + DCP value': {'derate': 5.0, 'range': [4.1, 10.3],
    'source_rows': ['gap512:FAMILY DSA q path+sparse MLA+DCP value (4.12-6.04)', 'decode:prof1b MlaQuery (5.14), sparse FMHA (10.27)', 'ops1:sparse MLA decode tw 5.1 (re-based)']},
  'DSA indexer logits': {'derate': 2.3, 'range': [2.2, 2.84], 'source_rows': ['gap512:FAMILY indexer logits']},
  'DSA exact top-k incl. DCP merge (P8)': {'derate': 11.0, 'range': [4.6, 18.9],
    'source_rows': ['gap512:FAMILY DSA DCP exact top-k (9.6-13.6)', 'decode:prof1b paged_mqa_logits+topk (4.63 r8, 18.9 r1)', 'decode:prof3 topk_merge+DCP value (3.92)']},
  'small unfused elementwise (AttnRes mix, SiLU, latent RMSNorm, router top-k, MoE prep)': {'derate': None, 'absolute_ms_per_step': 1.55, 'range': [10.7, 29.6],
    'note': 'flat ~1.53-1.58 ms/step on D3 B32-B92: model as an absolute floor, not a multiplier', 'source_rows': ['gap512:FAMILY small elementwise']},
  'AttnRes (D1 sdec cluster kernel)': {'derate': 2.02, 'range': [2.02, 2.02], 'source_rows': ['decode:prof1b attn_res cluster kernel']},
  'drafter + final (DSpark k=3)': {'derate': 2.3, 'range': [2.04, 3.38], 'source_rows': ['gap512:FAMILY drafter+final']},
  'SP comm AG/RS (in-step)': {'derate': 1.4, 'range': [1.2, 2.3], 'source_rows': ['gap512:FAMILY SP comm (1.20-1.48)', 'decode:prof3 SpAllGather+SpReduceScatter (1.59 r8, 2.30 r1)']},
  'DCP value exchange / o_proj+AR (D1)': {'derate': 1.25, 'range': [1.25, 1.26], 'source_rows': ['decode:prof1b DCP value (1.25)', 'decode:prof1b o-proj+AR (1.26)']},
 },
 'prefill': {
  'experts (MegaMoE)': {'derate': 1.0, 'range': [0.85, 1.64],
    'note': 'in-step vendored 0.97-1.05/layer (P2 C512); FI 6e438852b standalone 0.85-1.02 vs ops+comm (overlap; model sums); vs dedup model 1.09-1.14; P1 vendored in-step 1.64',
    'source_rows': ['gap512:MOE.megamoe per layer pf (0.97-1.05)', 'moe:EP16 MX t2048-16384 (0.85-0.91)', 'moe:EP8 MX t2048 (1.02)', 'moe:P6-16 dedup rows (1.09-1.11)', 'prefill:P1 MOE.megamoe (1.64)']},
  'MoE block incl. shared experts (critical path)': {'derate': 1.17, 'range': [0.88, 1.18], 'source_rows': ['prefill:block MoE (excl) P1 0.88, P2 1.18, P4-4 1.17, P3-16 1.15']},
  'shared experts (side stream)': {'derate': 3.9, 'range': [3.8, 4.0], 'note': 'shared_down 3.8-4.0x busy, shared_act 15-75x; contend with MegaMoE for SMs',
    'source_rows': ['prefill:MOE.shared_down', 'prefill:MOE.shared_act']},
  'large BF16 GEMMs (M>1024)': {'derate': 1.02, 'range': [0.96, 1.21], 'source_rows': ['prefill:per-op GEMM rows 0.96-1.09', 'ops1:GEMM BF16 M>1024 tw 1.19']},
  'MXFP8 GEMMs M>1024': {'derate': 1.25, 'range': [1.06, 1.40], 'source_rows': ['ops1:GEMM MXFP8 M>1024 tw']},
  'o_proj + RS/AR fused (gemm_rs_ar)': {'derate': 0.62, 'range': [0.55, 0.79], 'source_rows': ['prefill:KDA.o_proj+AR / DSA.o_proj+AR', 'gap512:fused o_proj+RS']},
  'KDA core prefill (FlashKDA, FP32 state)': {'derate': 1.8, 'range': [1.44, 4.19],
    'note': '96-head shape (P3-16/P9) falls to stock path: 2.4 in-step, 4.19 standalone', 'source_rows': ['prefill:KDA.core_prefill (1.44-2.41)', 'gap512:KDA core pf (2.07)', 'ops1:kda-077 (4.19), kda-051 (2.34)']},
  'KDA conv / o_norm / gate (unfused)': {'derate': 5.0, 'range': [2.0, 13.2], 'source_rows': ['prefill:KDA.conv (4.3-5.0)', 'prefill:KDA.o_norm (12.2-13.2)']},
  'DSA block incl. its comm (critical path)': {'derate': 1.55, 'range': [1.34, 1.81], 'source_rows': ['prefill:block DSA attention P1 1.34, P2 1.39, P4-4 1.73, P3-16 1.81']},
  'sparse MLA prefill (trtllm-gen decode-form, 128 padded heads)': {'derate': 3.5, 'range': [3.2, 3.97], 'source_rows': ['prefill:DSA.sparse_mla (3.20-3.35)', 'gap512:sparse MLA pf (3.3)', 'ops1:sparse_mla-089/094/161/162/310 (3.56-3.97, re-based)']},
  'DSA prologue (CuTe Kernel + ShardQ quant/insert + head pad)': {'derate': None, 'absolute_ms_per_step': [3.4, 8.5], 'range': [21, 53], 'source_rows': ['prefill:DSA.prologue+aux']},
  'DSA indexer logits': {'derate': 1.45, 'range': [1.38, 1.85], 'source_rows': ['prefill:DSA.idx_score', 'ops1:indexer prefill 1.85 (incl top-k)']},
  'DSA top-k (trtllm_gvr2 over ~78K logits)': {'derate': 4.5, 'range': [3.6, 7.0], 'source_rows': ['prefill:DSA.topk (3.6-5.6)', 'gap512:top-k pf (7)']},
  'AttnRes': {'derate': 1.3, 'range': [1.27, 1.33], 'note': 'P1 0.17 = model error (AttnRes on 1/8 rows + AG)', 'source_rows': ['prefill:block AttnRes']},
  'router / MoE top-k / latent norm (unfused)': {'derate': None, 'range': [2.2, 146], 'note': 'model assumes fused; charge extra memory pass', 'source_rows': ['prefill:MOE.router, MOE.topk, MOE.latent_norm']},
  'drafter (DSpark, incl NCCL AllGather)': {'derate': 7.0, 'range': [5.8, 8.8], 'source_rows': ['prefill:DS.drafter', 'gap512:drafter tail (AllGather 2.4 ms vs 20 us)']},
  'separate comm kernels (SP push / barrier)': {'derate': 1.3, 'range': [1.02, 1.72], 'source_rows': ['prefill:comm (separate kernels) P1 1.02, P2 1.72, P4-4 1.30']},
 },
 'comm_standalone (call-weighted, comm lane run7)': {
  'all-reduce (multimem)': {'derate': 0.99, 'range': [0.88, 2.42], 'note': 'large AR: model NVLS algbw too low (fit 0.64/0.77/0.84/0.88 TB/s N=2/4/8/16); small AR at ~5 us Lamport floor'},
  'reduce-scatter': {'derate': 2.02, 'range': [1.64, 2.70], 'note': 'RS ~0.75x AR cost, not 0.5x; comm-112 unfused torch.sum'},
  'P11 EP dispatch / P12 EP combine (standalone a2a)': {'derate': 1.24, 'range': [1.21, 1.67], 'note': 'do not stack on MegaMoE derate (fused in production)'},
  'P2 / P18 / P6 all-gather': {'derate': 1.33, 'range': [1.18, 2.04]},
  'P5 / P7b / P8 DCP-ShardQ exchanges': {'derate': 1.36, 'range': [1.29, 1.80]},
  'P13 DSA-DP / P14 front AG': {'derate': 1.33, 'range': [1.25, 1.57]},
  'P14 transpose / P16 / P19 (latency-bound)': {'derate': 1.67, 'range': [1.46, 2.68], 'note': 'floor 5.0-5.7 us (Lamport) vs model 2.86 us hop'},
  'P9 copy-engine context push': {'derate': 2.22, 'range': [1.86, 2.52]},
 },
}

out = {
 'generated': '2026-11-18',
 'scope': 'waves 1-2: EP <= 16 plus D3-32 (EP32, wave 2); model scope <= 32 GPUs; checkpoint mxfp4_release (MXFP8 MoE act, BF16 linears, FP8 MLA KV, MXFP4 idx KV, FP32 KDA state, exact top-k); decode k=3 L=135K, prefill F16K L78K',
 'definition': 'derate = measured_us / model t_ach_us (achievable column of out/optables.json / microbench/results/model/targets.json). >2.0 = KERNEL-DEBUG.',
 'versions': {
   'tree': 'dsrel/trees/rel-1767cf7 (SOURCE_REV 6a14e2fada)', 'flashinfer': '0.7.3.dev20261031 (in-step); MegaMoE standalone FI 6e438852b (wzhao/kimi-k3-megamoe-sm107-final-20261117)',
   'megamoe_in_step': 'vLLM-vendored megamoe_rubin / deep_gemm sm100_fp8_fp4_mega_moe_situ (older than FI 6e438852b)',
   'torch': '2.16.0.dev20261009+cu134 (ops1)', 'weights': 'dummy with real K3.1-Pro config (step benches); synthetic layout (MegaMoE)'},
 'by_family': fam,
 'by_topology': topo,
 'projection_unmeasured': proj,
 'review_caveats': [
   'ops1 unaligned-N KDA.in_proj flag is a bench artifact (release pads in_proj); rows re-based to padded measurement',
   'prefill/decode/gap512 in-step MegaMoE is the older vendored kernel; FI 6e438852b projects -21 ms/P1 step, -1.6..-2.5 ms/D3 step (projection)',
   'decode D2 ran MoE-TP8, model D2 is EP8',
   'comm P11/P12 standalone a2a overlap with MegaMoE in production; do not add on top of moe derate',
   'gap512 20-3400x derates on near-zero model terms = unfused op the model does not charge',
 ],
}
# ---------------- current-model achievable for every measured step point (calibrate.py basis) ----------------
import re, sys
sys.path.insert(0, PM)
import decode as _Dec, prefill as _Pre  # noqa: E402
from topologies import ALL as _ALL  # noqa: E402
for tp, e in topo.items():
    for p in e['points']:
        if e['phase'] == 'decode':
            a = _Dec.run_point(_ALL[tp], 'mxfp4_release', 'achievable', p['r_per_gpu'], p['k'])['t_ms']
        else:
            m = re.search(r'model \S+ F(\d+)(?: f(\d+))?', p['case'])
            if not m or 'L78k' not in p['case'] and 'fresh' not in p['case']:
                continue
            F = int(m.group(1)); f = int(m.group(2)) if m.group(2) else _Pre.F_REQ
            L = _Pre.L_REF if 'L78k' in p['case'] else F
            act = 'nvfp4' if p['moe_quant'].startswith('nvfp4') else None
            a = _Pre.run_point(_ALL[tp], 'mxfp4_release', 'achievable', F, L, f_req=min(f, F), moe_act=act)['t_ms']
        p['model_ach_ms_current'] = round(a, 3)
        p['derate_current'] = round(p['measured_ms'] / a, 3)
out['current_model_note'] = ('model_ach_ms / derate: wave-1 basis (targets.json, pre-fix model); model_ach_ms_current / '
                             'derate_current: the current model (achievable column), recomputed by this script; '
                             'NVFP4-MoE (release) prefill points use the O4 NVFP4-activation model')
json.dump(out, open(PM + '/inputs/derating.json', 'w'), indent=1)

# ---------- print tables for RESULTS.md ----------
for tp in ['D1', 'D2', 'D3']:
    print(tp, topo[tp]['derate_by_band'])
for band, rows in proj.items():
    print('==', band)
    for tp, x in rows.items():
        print('  ', tp, x)
