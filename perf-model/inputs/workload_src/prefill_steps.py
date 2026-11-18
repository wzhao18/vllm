import gzip,glob,os,re,sys,json
runs=sys.argv[1]; out={}
pat=re.compile(r'(\d\d-\d\d \d\d:\d\d:\d\d) \[context.py:894\] DCP ShardQ prefill .*?: (\d+)/(\d+) steps, (\d+)/(\d+) tokens')
pmnbt=re.compile(r'max_num_batched_tokens=(\d+)')
pkv=re.compile(r'GPU KV cache size: ([\d,]+) tokens')
pav=re.compile(r'Available KV cache memory: ([\d.]+) GiB')
for d in sorted(glob.glob(runs+'/*/agentx/logs')):
    run=d.split('/')[-3]; res={}
    for f in sorted(glob.glob(d+'/*_prefill_w*.out*'))+sorted(glob.glob(d+'/*_decode_w*.out*')):
        role='prefill' if 'prefill' in f else 'decode'
        op=gzip.open if f.endswith('.gz') else open
        steps=[];mn=None;kv=None;av=None;worker=None
        with op(f,'rt',errors='ignore') as fh:
            for line in fh:
                if 'context.py:894' in line and 'Worker_DP' in line:
                    w=re.search(r'\((Worker_\S+)',line).group(1)
                    if worker is None: worker=w
                    if w!=worker: continue
                    m=pat.search(line)
                    if m: steps.append((int(m.group(3)),int(m.group(5))))
                elif 'max_num_batched_tokens=' in line and mn is None:
                    mn=int(pmnbt.search(line).group(1))
                elif 'GPU KV cache size' in line and kv is None:
                    kv=int(pkv.search(line).group(1).replace(',',''))
                elif 'Available KV cache memory' in line and av is None:
                    av=float(pav.search(line).group(1))
        S=sum(s for s,t in steps); T=sum(t for s,t in steps)
        # windows with >=1 step; skip first windows (warmup) -> skip windows before half of max? keep all
        busy=[t/s for s,t in steps if s>0]
        busy.sort()
        res[os.path.basename(f)]=dict(role=role,max_num_batched_tokens=mn,gpu_kv_cache_tokens=kv,avail_kv_gib=av,
            n_windows=len(steps),steps=S,tokens=T,mean_tokens_per_step=(T/S if S else None),
            window_tokens_per_step_p10=busy[len(busy)//10] if busy else None,
            window_tokens_per_step_p50=busy[len(busy)//2] if busy else None,
            window_tokens_per_step_p90=busy[len(busy)*9//10] if busy else None,
            steps_per_10s_mean=(S/len(steps) if steps else None))
    out[run]=res
    for k,v in res.items():
        print(run,k,v['role'],v['max_num_batched_tokens'],v['gpu_kv_cache_tokens'],v['avail_kv_gib'],v['steps'],v['tokens'],
              v['mean_tokens_per_step'] and round(v['mean_tokens_per_step']),v['window_tokens_per_step_p50'] and round(v['window_tokens_per_step_p50']),v['steps_per_10s_mean'] and round(v['steps_per_10s_mean'],1))
json.dump(out,open(sys.argv[2],'w'),indent=1)
