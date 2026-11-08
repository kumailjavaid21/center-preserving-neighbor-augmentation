import math
import numpy as np
import torch
from torch.nn import functional as F
def blend_window(size=128):
    # Strictly positive raised-cosine window, endpoints >=1e-3 before outer product.
    t = np.hanning(size).astype(np.float32)
    t = np.maximum(t, 1e-3)
    return np.outer(t, t)


def geometry(shape):
    h,w=shape; hp=max(128,128+math.ceil(max(0,h-128)/64)*64); wp=max(128,128+math.ceil(max(0,w-128)/64)*64)
    positions=[(r,c) for r in range(0,hp-127,64) for c in range(0,wp-127,64)]
    return {'shape':(h,w),'padded_shape':(hp,wp),'positions':positions,'tiles':len(positions)}

def blend_tiles(tiles,layout):
    """FP32 tile products round BEFORE FP64 accumulation, exactly as NumPy tiler.

    Float64 tiles are supported for independent smooth finite-difference tests.
    All additions are out-of-place so overlapping paths remain differentiable.
    """
    assert len(tiles)==layout['tiles']
    hp,wp=layout['padded_shape']; h,w=layout['shape']
    out=torch.zeros((hp,wp),device=tiles.device,dtype=torch.float64); weights=torch.zeros_like(out)
    window=torch.as_tensor(blend_window(128),device=tiles.device,dtype=tiles.dtype)
    for tile,(r,c) in zip(tiles,layout['positions']):
        padding=(c,wp-c-128,r,hp-r-128)
        weighted=(window*tile).to(torch.float64)
        out=out+F.pad(weighted,padding)
        weights=weights+F.pad(window.to(torch.float64),padding)
    assert bool(torch.all(weights>0))
    return (out/weights)[:h,:w].to(tiles.dtype)

def tile_batches(y,layout):
    h,w=y.shape;hp,wp=layout['padded_shape'];padded=np.pad(y,((0,hp-h),(0,wp-w)),mode='edge')
    for start in range(0,layout['tiles'],4):yield torch.from_numpy(np.stack([padded[r:r+128,c:c+128] for r,c in layout['positions'][start:start+4]])[:,None].copy()).cuda()
