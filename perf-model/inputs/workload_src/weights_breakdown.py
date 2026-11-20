import json,re,sys,collections
d=json.load(open(sys.argv[1]))
L=collections.defaultdict(set)
for k in d:
    m=re.match(r'language_model\.model\.layers\.(\d+)\.(.*)',k)
    if m: L[int(m.group(1))].add(re.sub(r'experts\.\d+\.','experts.E.',m.group(2)))
idx=sorted(L); print('layer idx range',idx[0],idx[-1],len(idx))
dsa=[i for i in idx if any('indexer' in n for n in L[i])]
kda=[i for i in idx if any('A_log' in n for n in L[i])]
dense=[i for i in idx if any(n.startswith('mlp.') for n in L[i])]
print('dsa(0-based)',dsa); print('n kda',len(kda),'dense',dense)
# per layer-type tensor list for one representative
for name,i in (('dsa',dsa[0]),('kda',kda[1]),('dense',dense[0])):
    print('==',name,i)
    for n in sorted(L[i]):
        if 'experts.E.' in n and not n.startswith('block_sparse_moe.experts.E.w1'): 
            pass
        key=f'language_model.model.layers.{i}.'+n.replace('experts.E.','experts.0.')
        print('  ',n,d[key][0],d[key][1],d[key][2])
# categorize bytes
cat=collections.Counter(); cnt=collections.Counter()
def c(k):
    if k.startswith('vision_tower') or k.startswith('mm_projector'): return 'vision(unused for text)'
    if 'embed_tokens' in k: return 'embed_tokens(bf16)'
    if 'lm_head' in k: return 'lm_head(bf16)'
    m=re.match(r'language_model\.model\.layers\.(\d+)\.(.*)',k)
    if not m: return 'final_norm/res(bf16)'
    i=int(m.group(1)); n=m.group(2)
    lt='dsa' if i in dsa else 'kda'
    if '.experts.' in n: return 'moe_routed_experts_mxfp4(packed+scales)'
    if 'routed_expert_' in n: return 'moe_latent_up_down_norm(bf16)'
    if 'shared_experts' in n: return 'moe_shared_experts(bf16)'
    if 'gate.' in n and 'block_sparse_moe' in n: return 'moe_router(bf16+f32 bias)'
    if n.startswith('mlp.'): return 'dense_mlp_layer(bf16)'
    if 'indexer' in n: return 'dsa_indexer(bf16)'
    if n.startswith('self_attn'): return f'{lt}_attn(bf16/f32)'
    return 'norms_res_proj(bf16)'
for k,(dt,sh,b) in d.items():
    cat[c(k)]+=b; cnt[c(k)]+=1
tot=sum(cat.values())
for k,v in sorted(cat.items(),key=lambda x:-x[1]): print(f'{k:45s} {v/1e9:10.3f} GB  {cnt[k]}')
print('total',tot/1e9)
json.dump({'by_category_bytes':cat,'dsa_layers_0based':dsa,'kda_layers_0based':kda,'dense_layers_0based':dense},open(sys.argv[2],'w'),indent=1)
