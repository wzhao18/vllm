import os
HERE=os.path.dirname(os.path.abspath(__file__))
# Screening estimates for proposals.md (NOT the SOL model). Derated/measured rates only (13.9 TB/s HBM, 1.25 TB/s NVLink,
# 3.65/10.9/21.1 PF, 130 TF simt, 2.86 us hop, 0.78 us graph node). t_step = max(compute, memory) + comm/BW + latency floor.
import json,math
d=json.load(open(os.path.join(HERE,'..','inputs','ops.json')))
M=d['decode_per_request_marginal']
GB=1e9; HBM=13.9e12; NVL=1.25e12; HOP=2.86e-6; NODE=0.78e-6
RATE={'bf16':3.65e15,'fp8':10.9e15,'fp4':21.1e15,'simt':130e12}
SPEC={'bf16':4.0e15,'fp8':17.5e15,'fp4':35e15,'simt':130e12}
HBM_FREE=272.48*2**30/GB; RESERVE=20.0
CK={ # weights GB by family
 'mx':dict(kda=61.26,kda_dt='bf16',dsa=11.51,moe_rep=10.64,shared=24.31,sh_dt='bf16',dense0=1.45,emb=2.35,head=2.35,exp=1446.46,exp_dt='fp8',disp=3696),
 'nv':dict(kda=31.59,kda_dt='fp8',dsa=11.51,moe_rep=10.64,shared=12.53,sh_dt='fp8',dense0=1.45,emb=2.35,head=2.35,exp=1531.5,exp_dt='fp4',disp=2016),
}
DRAFT=4.96
ops={o['id']:o for o in d['ops']}
def dt(i,c):
    cc=ops[i]['dtype']['compute']
    if i.startswith('KDA.') and i.endswith('proj') and i!='KDA.f_b_proj': return c['kda_dt']
    if i in('MOE.shared_gate_up','MOE.shared_down'): return c['sh_dt']
    if i in('MOE.fc1','MOE.fc2'): return c['exp_dt']
    if 'mxf4' in cc: return 'fp4'
    if 'bf16' in cc: return 'bf16'
    return 'simt'
REPL_DSA={'DSA.idx_score','DSA.topk','DSA.sparse_mla','DSA.prologue','DSA.qkv_a_packed'}
def per_req(k,c,dsa_rep):
    v=M[f'k{k}_L135K']; b=0; t=0
    for i,x in v.items():
        f=dsa_rep if i in REPL_DSA else 1
        b+=x['bytes']*f; t+=x['flops']*f/RATE[dt(i,c)]
    return b,t
def f_touch(routes): return 1-(1-1/896)**routes
STATE135=2.09+0.495+0.047+0.027; STATE189=2.09*189/135+0.495+0.047+0.027
def topo(name,N,dsa,moe_dense,E,moe='ep',dsa_w=None):
    return dict(name=name,N=N,dsa=dsa,moe_dense=moe_dense,E=E,moe=moe,dsa_w=dsa_w)
T=[
 topo('D1 TP16+DCP16, MoE-TP16',16,'dcp','tp',16,moe='tp'),
 topo('D2 TP8+DCP8, EP8 token-shard',8,'dcp','tp',8),
 topo('D3 DP4x(TP4=DCP4) SP, EP16',4,'dcp','rep',16),
 topo('D4 DEP16',1,'own','rep',16),
 topo('D4 DEP32',1,'own','rep',32),
 topo('D4 DEP72',1,'own','rep',72),
 topo('D5 TP4 DSA-DP SP, EP16',4,'dsadp','rep',16),
 topo('D6 TP4 plain (rep DSA) SP, EP16',4,'rep','rep',16),
 topo('D7 TP4 DSA-DP, MoE-dense TP4, EP64',4,'dsadp','tp',64),
 topo('D7 TP2 DSA-DP, MoE-dense TP2, EP64',2,'dsadp','tp',64),
 topo('D7 TP8 DSA-DP, MoE-dense TP8, EP64',8,'dsadp','tp',64),
 topo('D8 TP16+DCP16, MoE-dense TP16, EP64',16,'dcp','tp',64),
 topo('D9 DSA-CP: DEP + DCP16 on DSA only, EP64',1,'dsacp','rep',64,dsa_w=16),
]
def weights(t,c):
    N=t['N']; dsaN=1 if t['dsa'] in('dsadp','own','dsacp') else N
    w=c['kda']/N + c['dsa']/dsaN
    if t['moe_dense']=='tp': w+=(c['moe_rep']+c['shared']+c['dense0'])/N
    else: w+=c['moe_rep']+c['shared']+c['dense0']
    w+=(c['emb']+c['head'])/N + c['exp']/t['E'] + DRAFT/max(1,min(N,16))
    return w
def state_per_gpu_req(t,ctx):
    s=STATE135 if ctx==135 else STATE189
    kv=2.09*(1 if ctx==135 else 189/135)
    rep = t['N'] if t['dsa']=='rep' else 1
    return (s + kv*(rep-1))/t['N']
def step(t,c,r,k=3):
    q=k+1; N=t['N']; own=r*q
    dense=weights(t,c)-c['exp']/t['E']
    group_rows=own*N
    if t['moe']=='tp': expb=c['exp']/N*f_touch(group_rows*16)
    else: expb=c['exp']/t['E']*f_touch(own*t['E']*16)
    rep = N if t['dsa']=='rep' else 1
    b_req,t_req=per_req(k,c,rep)
    mem=(dense+expb)*GB + r*b_req
    comp=r*t_req   # per-request work is spread over the group, r requests per GPU
    # comm bytes in per GPU (per own row per step)
    h2=14336; cb=0; hops=0
    nattn=93
    if N>1:
        if t['dsa'] in('dsadp',): cb+=69*2*(N-1)*h2 + 24*2*h2*(N-1)/N; hops+=93*2
        else: cb+=nattn*2*(N-1)*h2; hops+=93*2
    if t['dsa']=='dcp': cb+=24*(N-1)*(96*836+1536); hops+=24*3
    if t['dsa']=='dsacp':
        W=t['dsa_w']; cb+=24*(W-1)*(96*836+1536); hops+=24*3
    if t['moe']=='tp': cb+=92*(N-1)*53800; hops+=92*3
    else:
        if t['moe_dense']=='tp' and N>1: cb+=92*2*(N-1)*h2; hops+=92*2
        E=t['E']; cb+=92*(c['disp']+7168)*16*(E-1)/E; hops+=92*3
    comm=own*cb/NVL
    lat=hops*HOP + 93*10*NODE
    # drafter: weights /N (TP<=16), 11 AR sites + k Markov hops
    dN=max(1,min(N,16)); draft=DRAFT*GB/dN/HBM + (11+k)*HOP*(N>1) + 6*NODE*5
    t_main=max(mem/HBM,comp)+comm+lat
    return t_main+draft, mem/HBM, comp, comm, lat
AL={3:3.08,5:3.81,7:4.16}
out=[]
for ck in('mx','nv'):
  c=CK[ck]
  print('\n== checkpoint',ck)
  print('%-44s %7s %6s %6s | '%('topology','W/GPU','rmax135','rmax189')+' '.join('r=%-5s'%r for r in(0.25,1,4,16,32,64)))
  for t in T:
    w=weights(t,c); free=HBM_FREE-w-RESERVE
    r1=free/state_per_gpu_req(t,135); r2=free/state_per_gpu_req(t,189)
    cells=[]
    for r in(0.25,1,4,16,32,64):
        if r>r1: cells.append('  cap  '); continue
        s=step(t,c,r)[0]; cells.append('%5.1f/%-4.0f'%(s*1e3, r*AL[3]/s))
    print('%-44s %6.0f %6.0f %6.0f | '%(t['name'],w,max(r1,0),max(r2,0))+' '.join(cells))
    out.append((ck,t['name'],w,r1,r2,[(r,)+step(t,c,r) for r in(0.25,1,4,16,32,64)]))
json.dump(out,open(os.path.join(HERE,'screen_decode.json'),'w'))
# breakdown at r=16 for a few
for ck in('mx','nv'):
  for t in (T[2],T[3],T[6],T[8],T[0]):
    s=step(t,CK[ck],16); print(ck,t['name'],'r16 step %.1f mem %.1f comp %.1f comm %.1f lat %.2f'%tuple(x*1e3 for x in s))
