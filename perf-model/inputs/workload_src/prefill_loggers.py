import gzip,glob,re,sys,json
pat=re.compile(r'(\d\d-\d\d \d\d:\d\d:\d\d) \[loggers.py:335\] Engine (\d+): Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs.*?Prefix cache hit rate: ([\d.]+)%, External prefix cache hit rate: ([\d.]+)%')
out={}
for d in sorted(glob.glob(sys.argv[1]+'/*/agentx/logs')):
    run=d.split('/')[-3]; r={}
    for f in sorted(glob.glob(d+'/*_prefill_w*.out*')):
        op=gzip.open if f.endswith('.gz') else open
        rows=[]
        with op(f,'rt',errors='ignore') as fh:
            for line in fh:
                if 'loggers.py:335' in line:
                    m=pat.search(line)
                    if m: rows.append((float(m.group(3)),int(m.group(5)),int(m.group(6)),float(m.group(7)),float(m.group(8))))
        if not rows: continue
        # steady state: drop first 25% (primer/warmup) 
        ss=rows[len(rows)//4:]
        pt=[x[0] for x in ss]
        r[f.split('/')[-1]]=dict(n_windows=len(rows),ss_mean_prompt_tps=sum(pt)/len(pt),ss_max_prompt_tps=max(pt),
            ss_mean_running=sum(x[1] for x in ss)/len(ss),ss_mean_waiting=sum(x[2] for x in ss)/len(ss),
            ss_mean_local_hit_pct=sum(x[3] for x in ss)/len(ss),ss_mean_ext_hit_pct=sum(x[4] for x in ss)/len(ss))
    out[run]=r
    for k,v in r.items(): print(run,k,{a:round(b,1) for a,b in v.items()})
json.dump(out,open(sys.argv[2],'w'),indent=1)
