"""Original canonical float32 references and genuine adjacent acquired planes."""
from pathlib import Path
import functools,math
import numpy as np
import nibabel as nib
from src.io import ROOT,read,save

def prepare_subject(path,record,output):
    image=nib.as_closest_canonical(nib.load(path))
    volume=image.get_fdata(dtype=np.float32)
    if volume.ndim!=3 or min(volume.shape)<12 or not np.isfinite(volume).all():raise ValueError('Ineligible canonical volume')
    nonzero=volume[volume!=0]
    factor=float(np.percentile(nonzero,99.5)) if len(nonzero) else 0
    if factor<=0:raise ValueError('Invalid reference normalization')
    z=volume.shape[2]
    indices=np.rint(np.linspace(math.ceil(.2*(z-1)),math.floor(.8*(z-1)),12)).astype(int)
    if len(set(indices))!=12 or any(not 0<k<z-1 for k in indices):raise ValueError('Original indices lack genuine adjacent planes')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pairs=[]
    for position,index in enumerate(indices):
        ref=np.clip(volume[:,:,index].T/factor,0,1).astype(np.float32)
        np.save(output/f'{position:02d}.npy',ref,allow_pickle=False)
        pairs.append([np.clip(volume[:,:,j].T/factor,0,1).astype(np.float32) for j in (index-1,index+1)])
    np.save(output/'neighbors.npy',np.stack(pairs),allow_pickle=False)
    meta={**record,'slice_indices':indices.tolist(),'canonical_shape':list(volume.shape),
          'canonical_affine':image.affine.tolist(),'canonical_axcodes':list(nib.aff2axcodes(image.affine)),
          'nonzero_995_percentile':factor,'slice_shape':[volume.shape[1],volume.shape[0]]}
    save(output/'metadata.json',meta)
    return meta

class PreparedData:
    def __init__(self,directory):
        self.directory=Path(directory)
        population=read(ROOT/'metadata/development_population.json')
        self.populations={x['subject']:{**x,'split':split} for key,split in [('train','train'),('development','validation')] for x in population[key]}
        assert sum(x['split']=='train' for x in self.populations.values())==64 and sum(x['split']=='validation' for x in self.populations.values())==16
    def folder(self,sid):
        if sid not in self.populations:raise ValueError('Subject is outside published TRAIN/development population')
        return self.directory/sid
    @functools.lru_cache(maxsize=80)
    def metadata(self,sid):
        value=read(self.folder(sid)/'metadata.json')
        assert value['subject']==sid and value['site']==self.populations[sid]['site'] and value['split']==self.populations[sid]['split']
        return value
    @functools.lru_cache(maxsize=960)
    def reference(self,sid,pos):return np.load(self.folder(sid)/f'{pos:02d}.npy',allow_pickle=False)
    @functools.lru_cache(maxsize=80)
    def neighbors(self,sid):return np.load(self.folder(sid)/'neighbors.npy',mmap_mode='r',allow_pickle=False)
