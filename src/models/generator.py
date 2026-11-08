import torch
from torch import nn
from .nafnet import NAFNet

class FourNAF(NAFNet):
 def __init__(self):
  super().__init__(activation_checkpointing=False);self.intro=nn.Conv2d(4,32,3,padding=1)
 def forward(self,inp):
  assert inp.ndim==4 and inp.shape[1]==4
  _,_,H,W=inp.shape;x=self.intro(self.check_image_size(inp));encs=[]
  for encoder,down in zip(self.encoders,self.downs):x=self._run_blocks(encoder,x);encs.append(x);x=down(x)
  x=self._run_blocks(self.middle_blks,x)
  for decoder,up,skip in zip(self.decoders,self.ups,encs[::-1]):x=up(x);x=x+skip;x=self._run_blocks(decoder,x)
  return self.ending(x)[:,:,:H,:W]

class Generator(nn.Module):
 def __init__(self):super().__init__();self.backbone=FourNAF();self.method='NAF15-U';self._G_microbatch=4
 def forward(self,y,sigma,minus,plus):return y+self.backbone(torch.cat((y,sigma,minus,plus),1))
