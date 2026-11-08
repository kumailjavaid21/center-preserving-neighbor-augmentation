import numpy as np
import torch
from torch.nn import functional as F
from .native_tiling import blend_tiles
from .native_tiling import geometry,tile_batches
ARMS=("CENTER","PLAIN3","COMPAT3")
@torch.no_grad()
def prepare_tiles(n,y,force_zero=False):
    layout=geometry(y.shape);tiles=[];sigmas=[]
    # N retains the original microbatch-four forward, independent of the
    # memory-limited one-tile NAFNet forward/backward microbatch.
    for batch in tile_batches(y,layout):
        sigma=torch.zeros_like(batch) if force_zero else n(batch)
        for index in range(len(batch)):
            tiles.append(batch[index:index+1].contiguous());sigmas.append(sigma[index:index+1].contiguous())
    return tiles,sigmas,layout


def gaussian(like):
 x=torch.arange(-2,3,device=like.device,dtype=like.dtype);a=torch.exp(-.5*x.square());h=a[:,None]*a[None,:];return (h/h.sum())[None,None]

def moments(y,s):
 u=y.square()-2*s.square();v=4*s.square()*u.clamp_min(0)+4*s.pow(4);h=gaussian(y)
 return F.conv2d(u[None,None],h)[0,0],F.conv2d(v[None,None],h.square())[0,0]

def compatibility(y0,s0,yj,sj):
 m0,v0=moments(y0,s0);mj,vj=moments(yj,sj);q=(mj-m0).square()/(vj+v0+1e-8);interior=1/(1+(q-1).clamp_min(0));g=F.pad(interior,(2,2,2,2),value=1.)
 assert torch.isfinite(g).all() and g.min()>0 and g.max()<=1
 return g

@torch.no_grad()
def features(n,y0,y_minus,y_plus,arm):
 assert arm in ARMS and y0.shape==y_minus.shape==y_plus.shape
 tiles,sigmas,layout=prepare_tiles(n,y0);s0=blend_tiles(torch.cat([s[:,0] for s in sigmas]),layout);y=torch.as_tensor(y0,device='cuda');aux=[];gates=[]
 for observation in (y_minus,y_plus):
  j=torch.as_tensor(observation,device='cuda')
  if arm=='CENTER':a=torch.zeros_like(y)
  elif arm=='PLAIN3':a=j-y
  else:
   _,ss,lj=prepare_tiles(n,observation);assert lj==layout;sj=blend_tiles(torch.cat([s[:,0] for s in ss]),lj);gate=compatibility(y,s0,j,sj);gates.append(gate);a=gate*(j-y)
  hp,wp=layout['padded_shape'];h,w=y0.shape;z=F.pad(a[None,None],(0,wp-w,0,hp-h),mode='replicate');aux.append([z[:,:,r:r+128,c:c+128].contiguous() for r,c in layout['positions']])
 return tiles,sigmas,layout,s0,aux,gates

def native(g,n,y0,y_minus,y_plus,arm):
 tiles,sigmas,layout,s0,aux,gates=features(n,y0,y_minus,y_plus,arm);out=[]
 for i in range(0,len(tiles),4):out.append(g(*[torch.cat(v[i:i+4]) for v in (tiles,sigmas,aux[0],aux[1])])[:,0])
 p=blend_tiles(torch.cat(out),layout);assert torch.isfinite(p).all()
 diag={'tiles':layout['tiles'],'G_native_passes':1,'G_tile_evaluations':layout['tiles'],'G_calls':len(out),'N_native_passes':3 if arm=='COMPAT3' else 1,'N_tile_evaluations':layout['tiles']*(3 if arm=='COMPAT3' else 1),'N_calls':len(out)*(3 if arm=='COMPAT3' else 1),'G_microbatch':4,'N_microbatch':4}
 if gates:diag.update(gate_mean=float(torch.stack(gates).mean()),gate_below_half=float((torch.stack(gates)<.5).float().mean()),gate_exact_one=float((torch.stack(gates)==1).float().mean()),gate_min=float(torch.stack(gates).min()))
 return p,s0,diag
