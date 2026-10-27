"""Синтетические «голоса» для автотестов (нужен scipy). Не часть лаборатории."""
import numpy as np
from scipy.signal import lfilter
SR=16000
def resonator(f,bw):
    r=np.exp(-np.pi*bw/SR); th=2*np.pi*f/SR
    return [1.0],[1,-2*r*np.cos(th),r*r]
VOW=[(700,1220,2600),(300,2300,3000),(350,800,2500),(500,1500,2500)]  # a, i, u, e-ish
def voice(f0,fscale,dur=2.0,seed=0,jit=0.01,breath=0.01):
    rng=np.random.default_rng(seed)
    n=int(SR*dur); out=np.zeros(n)
    segs=np.array_split(np.arange(n),8)
    for k,sg in enumerate(segs):
        v=VOW[rng.integers(0,len(VOW))]
        # glottal pulses
        e=np.zeros(len(sg)); t=0.0
        while t<len(sg):
            e[int(t)]=1.0
            t+=SR/(f0*(1+jit*rng.standard_normal())*(1+0.1*np.sin(k)))
        y=e
        for F,bw in zip(v,(80,100,120)):
            b,a=resonator(F*fscale,bw); y=lfilter(b,a,y)
        out[sg]=y*np.hanning(len(sg))**0.3
    out=out/np.abs(out).max()
    out+=breath*rng.standard_normal(n)
    return out.astype(np.float32)
