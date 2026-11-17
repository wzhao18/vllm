import math
OSL=1030
def hyb_step(Br,s=0.95,slope=0.30): return s*(32.43+slope*(Br-52.3))
def tp16_step(B,s=0.94): return s*(4.811+0.1085*B+1.024*math.sqrt(B))
def tp8_ttft(r):
    rho=r/7.20; p90=0.32+0.871*rho/(1-rho) if rho<1 else 99
    rhom=r/7.35; mean=0.22+0.506*rhom/(1-rhom)
    return mean,p90
def hyb_ttft(r_eng,rmax=5.44,resid=1.83):
    rho=min(r_eng/rmax,0.98); inpf=0.3+0.51*rho/(1-rho)
    mean=inpf+resid; return mean, mean*6.43/4.16
def solve(C,nD,dec,Z,ttftf,nPeng,gpus,ISLOSL,AL=3.08,p90r=1.105,ndp=4):
    R=C/30
    for _ in range(300):
        Bi=1.04*R*OSL*0.01/nD
        for _ in range(50):
            itl=(dec(Bi/ndp) if ndp>1 else dec(Bi))/AL
            Bi=0.5*Bi+0.5*1.04*(R/nD)*OSL*itl/1000
        tm,tp=ttftf(R/nPeng)
        Rn=C/(Z+tm+OSL*itl/1000); R=0.7*R+0.3*Rn
    return dict(C=C,R=round(R,2),B_inst=round(Bi,1),B_rank=round(Bi/ndp,1),itl=round(itl,2),x=round(1000/(itl*p90r),1),y=round(R*ISLOSL/gpus),ttft_mean=round(tm,2),ttft_p90=round(tp,2),r_eng=round(R/nPeng,2))
# validate frozen C512
print('frz C512',solve(512,1,lambda b:hyb_step(b,1.0),13.7,hyb_ttft,4,32,135.5e3,AL=2.87))
print('frz C256 2P1D',solve(256,1,lambda b:tp16_step(b,1.0),15.4,tp8_ttft,2,32,143.5e3,AL=2.87,p90r=1.18,ndp=1))
print('frz C160 1P1D',solve(160,1,lambda b:tp16_step(b,1.0),17.0,tp8_ttft,1,24,145.6e3,AL=2.87,p90r=1.17,ndp=1))
print('--- release expected front')
for C in (128,160): print('rel 1P1D TP',solve(C,1,tp16_step,17.0,tp8_ttft,1,24,145.6e3,p90r=1.17,ndp=1))
for C in (224,256): print('rel 2P1D TP',solve(C,1,tp16_step,15.6,tp8_ttft,2,32,143.5e3,p90r=1.18,ndp=1))
print('rel C512',solve(512,1,hyb_step,13.7,hyb_ttft,4,32,135.5e3))
for resid in (0.8,1.83):
  print('--- hybrid 1P1D lower C, resid',resid)
  for C,Z in ((256,15.4),(320,15.0),(384,14.5),(448,14.0)):
    print(solve(C,1,hyb_step,Z,lambda r:hyb_ttft(r,resid=resid),4,32,137e3))
print('--- 1.5:1 hybrid P 6 engines + D DP4, 40 GPUs')
for resid in (0.8,1.83):
  for C in (576,640,704):
    print(resid,solve(C,1,hyb_step,13.7,lambda r:hyb_ttft(r,resid=resid),6,40,135.5e3))
print('--- slope 0.35 sens, s=0.985')
for C in (384,640,704):
    print(solve(C,1,lambda b:hyb_step(b,0.985,0.35),13.7 if C>500 else 14.5,hyb_ttft,6 if C>500 else 4,40 if C>500 else 32,135.5e3))
print('--- 2 hybrid P + 1 D, 48 GPUs')
for C in (640,704): print(solve(C,1,hyb_step,13.7,hyb_ttft,8,48,135.5e3))
print('--- 3P1D TP8 + TP16 D C288/320 40 GPUs')
for C in (288,320): print(solve(C,1,tp16_step,15.4,tp8_ttft,3,40,143.5e3,p90r=1.18,ndp=1))
print('--- extra')
for C in (192,208): print('rel 2P1D TP',solve(C,1,tp16_step,16.2,tp8_ttft,2,32,144e3,p90r=1.18,ndp=1))
print('rel 1P1D TP C144',solve(144,1,tp16_step,17.0,tp8_ttft,1,24,145.6e3,p90r=1.17,ndp=1))
print('hyb C256 Z16',solve(256,1,hyb_step,16.0,hyb_ttft,4,32,137e3))
