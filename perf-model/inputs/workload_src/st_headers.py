import json,struct,glob,os,sys
d=sys.argv[1]; out={}
for p in sorted(glob.glob(os.path.join(d,'model-*.safetensors'))):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
    for k,v in h.items():
        if k=='__metadata__': continue
        out[k]=(v['dtype'],v['shape'],v['data_offsets'][1]-v['data_offsets'][0])
json.dump(out,sys.stdout)
