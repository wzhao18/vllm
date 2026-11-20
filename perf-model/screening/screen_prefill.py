import os
HERE=os.path.dirname(os.path.abspath(__file__))
# Prefill screening (NOT the SOL model): per-GPU fresh tok/s at the 16K-chunk / 120K-prefix reference, both checkpoints.
import json
d=json.load(open(os.path.join(HERE,'..','inputs','ops.json')))
ref={e['id']:e for e in d['reference_evaluations']['prefill_C16384_P120000']}
ops={o['id']:o for o in d['ops']}
Tc=16384; NVL=1.25e12
RATES={'derated':({'bf16':3.65e15,'fp8':10.9e15,'fp4':21.1e15,'simt':130e12},13.9e12),
       'spec':({'bf16':4.0e15,'fp8':17.5e15,'fp4':35e15,'simt':130e12},22e12)}
CK={'mx':dict(kda_dt='bf16',sh_dt='bf16',exp_dt='fp8',disp=3696,exp=1446.46),
    'nv':dict(kda_dt='fp8',sh_dt='fp8',exp_dt='fp4',disp=2016,exp=1531.5)}
def dt(i,c):
    cc=ops[i]['dtype']['compute']
    if i.startswith('KDA.') and i.endswith('proj') and i!='KDA.f_b_proj': return c['kda_dt']
    if i in('MOE.shared_gate_up','MOE.shared_down'): return c['sh_dt']
    if i in('MOE.fc1','MOE.fc2'): return c['exp_dt']
    if 'mxf4' in cc: return 'fp4'
    if 'bf16' in cc: return 'bf16'
    return 'simt'
REP_PLAIN={'DSA.qkv_a_packed','DSA.prologue','DSA.idx_score','DSA.topk','DSA.idx_q_quant'}
REP_SHARDQ={'DSA.qkv_a_packed','DSA.prologue'}
SKIP={'L.sp_gather','F.sp_gather_out','MOE.dispatch','MOE.combine'}
def tok_time(c,rates,rep,N,E,pad=1.0):
    P,H=rates; t=0
    for i,e in ref.items():
        if i in SKIP: continue
        wb=e['weight_bytes']
        if i in('MOE.fc1','MOE.fc2'): wb=wb/ max(E,1)   # each GPU streams only its experts once per step
        x=max(e['flops']/P[dt(i,c)], (wb+e['act_bytes']+e['kv_bytes']+e['state_bytes'])/H)
        if i=='DSA.sparse_mla': x*=pad
        t+= x if i in rep else x/N   # replicated ops cost full on every rank of the N group
    return t*N/Tc   # GPU-seconds per token / ... -> per-GPU time per own token
def comm_tok(kind,N,E,c,C=130000,T=16384):
    h2=14336; b=0
    if kind=='tp_ar': b+=186*N*h2                     # attn + MoE ARs, NVLS ~M in per GPU
    if kind in('sp','sp_dsadp'): b+= (69 if kind=='sp_dsadp' else 93)*2*(N-1)*h2
    if kind=='sp_dsadp': b+=24*2*2*h2*(N-1)/N
    if kind in('tp_ar','sp') and N>1:  # DCP ShardQ (context push via CE, q scatter, out gather)
        b+=24*((C/(T/N))*644*(N-1)/N + (N-1)/N*96*(192+128)*2)
    b+=92*(c['disp']+7168)*16*(E-1)/E
    return b
P=[('P1 TP8+DCP8 ShardQ, EP8 token-shard','tp_ar',8,8,REP_SHARDQ,1),
   ('P2 DP4x(TP4=DCP4) SP ShardQ, EP16','sp',4,16,REP_SHARDQ,1),
   ('P3 DEP16','dep',1,16,set(),1),('P3 DEP32','dep',1,32,set(),1),
   ('P4 TP4xDP DSA-DP SP, EP16','sp_dsadp',4,16,set(),1),('P4 TP2xDP DSA-DP SP, EP16','sp_dsadp',2,16,set(),1),
   ('P5 TP4 plain (rep DSA, padded heads) SP, EP16','sp',4,16,REP_PLAIN,64/24),
   ('P5 TP8 plain (rep DSA) token-shard, EP8','tp_ar',8,8,REP_PLAIN,64/12),
   ('P6 TP2 DSA-DP, MoE wide EP64','sp_dsadp',2,64,set(),1),
   ('P7 PP4 x DEP16 (stage = 23 layers)','dep',1,16,set(),1)]
for ck in('mx','nv'):
  c=CK[ck]; print('\n== ',ck)
  for name,kind,N,E,rep,pad in P:
    row=[]
    for rk in('spec','derated'):
      t=tok_time(c,RATES[rk],rep,N,E,pad)
      cm=comm_tok(kind if kind!='dep' else 'dep',N,E,c)/NVL
      row.append('%s comp %.0fK  +exposed %.0fK  overlapped %.0fK  (comm %.0f%% of NVLink at overlapped rate)'%(rk,1/t/1e3,1/(t+cm)/1e3,1/max(t,cm)/1e3,100*cm/max(t,cm)))
    print('%-46s'%name,' | '.join(row))
