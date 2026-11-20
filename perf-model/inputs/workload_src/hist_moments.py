import sys,json,glob,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from analyze_agentx import analyze
runs=sys.argv[1]; recs=[]
for p in sorted(glob.glob(os.path.join(runs,'*','agentx','logs','agentic','*','aiperf_artifacts','profile_export.jsonl*'))):
    r,_=analyze(p); recs+=r
edges=[0,1024,2048,4096,8192,16384,32768,65536,98304,131072,196608,262144,393216,524288,786432,1048576]
def hist(vals,w=None):
    w=w or [1]*len(vals); tot=sum(w); out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        out.append({'lo':lo,'hi':hi,'frac':sum(x for v,x in zip(vals,w) if lo<=v<hi)/tot})
    return out
isl=[r['isl'] for r in recs]; osl=[r['osl'] for r in recs]; cr=[r['cr'] for r in recs]
fr=[i-c for i,c in zip(isl,cr)]
n=len(recs)
res={
 'n_requests':n,
 'hist_isl_request_frac':hist(isl),
 'hist_fresh_request_frac':hist(fr),
 'hist_fresh_token_mass':hist(fr,fr),
 'hist_cached_prefix_fresh_token_weighted':hist(cr,fr),
 'hist_decode_ctx_osl_weighted':hist([i+o/2 for i,o in zip(isl,osl)],osl),
 'hist_osl_request_frac':[{'lo':lo,'hi':hi,'frac':sum(1 for o in osl if lo<=o<hi)/n} for lo,hi in zip([0,64,128,256,512,1024,2048,4096,8192,16384],[64,128,256,512,1024,2048,4096,8192,16384,10**6])],
 # per-request means of attention work terms (dense causal pairs)
 'mean_fresh':sum(fr)/n,
 'mean_fresh_x_cached':sum(f*c for f,c in zip(fr,cr))/n,
 'mean_fresh_sq_half':sum(f*f/2 for f in fr)/n,
 'mean_dense_attn_pairs_prefill':sum(f*c+f*(f+1)/2 for f,c in zip(fr,cr))/n,
 'mean_sparse_attn_pairs_prefill_topk2048':sum(sum(min(c+j,2048) for j in range(1,f+1)) if f<=4096 else (f*2048 - (max(0,2048-c)**2)/2 if c<2048 else f*2048) for f,c in zip(fr,cr))/n,
 'mean_osl':sum(osl)/n,
 'mean_decode_ctx_sum_per_request':sum(o*i+o*(o+1)/2 for i,o in zip(isl,osl))/n,
 'mean_isl':sum(isl)/n,
 'E_isl_plus_osl':sum(i+o for i,o in zip(isl,osl))/n,
}
res['decode_ctx_per_output_token_mean']=res['mean_decode_ctx_sum_per_request']/res['mean_osl']
res['dense_pairs_per_fresh_token']=res['mean_dense_attn_pairs_prefill']/res['mean_fresh']
json.dump(res,open(sys.argv[2],'w'),indent=1)
print(json.dumps({k:v for k,v in res.items() if not k.startswith('hist')},indent=1))
