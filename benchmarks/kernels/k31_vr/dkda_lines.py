import csv,sys,collections
num=lambda x: float(x.replace(",","")) if x.strip() not in ("","-") else 0.0
cur="";key=None;agg=collections.defaultdict(lambda:[0,0,""]);hdr=None
for r in csv.reader(open(sys.argv[1])):
    if not r: continue
    if r[0]=="File Path": cur=r[1].rsplit("/",1)[-1]; continue
    if r[0]=="Line No": hdr=r; ki=r.index("Warp Stall Sampling (All Samples)"); ii=r.index("Instructions Executed"); continue
    if hdr is None or len(r)<=ii: continue
    if r[0].strip().isdigit(): key=(cur,int(r[0])); agg[key][2]=r[1].strip()[:60]; continue
    if key and r[2].startswith("0x"): a=agg[key]; a[0]+=num(r[ii]); a[1]+=num(r[ki])
T=sum(v[0] for v in agg.values()); S=sum(v[1] for v in agg.values())
for k,v in sorted(agg.items(),key=lambda x:-x[1][int(sys.argv[3]) if len(sys.argv)>3 else 0])[:int(sys.argv[2])]:
    print("%-12s %4d i %4.1f%% s %4.1f%% %s"%(k[0][:12],k[1],100*v[0]/T,100*v[1]/S,v[2]))
