import numpy as np
from .randomness import named_seed
from .rician_corruption import sigma_field,rician
from .transforms import ah,transform
from .transforms import inverse
CONDITIONS=('uniform_003','uniform_006','uniform_009','spatial','clean')
METHOD_FAMILIES=('SHIFT','NOISE','SHIFT_AND_NOISE','CONTEXT_DROPOUT')
_source=None
def configure(source):
 global _source
 _source=source
def populations():return _source.populations
def metadata(sid):return _source.metadata(sid)
def reference(sid,pos):return _source.reference(sid,pos)
def neighbors(sid):return _source.neighbors(sid)


def shift(a, displacement):
 r,c=displacement;h,w=a.shape
 return np.ascontiguousarray(a[np.clip(np.arange(h)-r,0,h-1)[:,None],np.clip(np.arange(w)-c,0,w-1)[None,:]])

def training(seed,index,ci,purpose='scientific',subject=None,position=None):
 ids=sorted(s for s,r in populations().items() if r['split']=='train');assert len(ids)==64
 ns='research17_context_robustness_v1'
 selection=named_seed(ns,seed,index,CONDITIONS[ci],'subject_uniform',purpose)
 rng=np.random.default_rng(selection);sid=subject or ids[int(rng.integers(64))];pos=int(rng.integers(12)) if position is None else position;assert sid in ids
 k=metadata(sid)['slice_indices'][pos]
 def key(component):return named_seed(ns,seed,index,CONDITIONS[ci],sid,k,component,purpose)
 x=reference(sid,pos);pair=neighbors(sid)[pos]
 rng=np.random.default_rng(key('center'));sigma=sigma_field(x.shape,rng,CONDITIONS[ci]);y=rician(x,sigma,rng)
 nkeys=[key('minus_innovations'),key('plus_innovations')]
 obs=[rician(a,sigma,np.random.default_rng(nk)) for a,nk in zip(pair,nkeys)]
 ar=np.random.default_rng(key('common_augmentation'));augment=purpose in ('scientific','qualification')
 m={'seed':seed,'index':index,'condition_index':ci,'condition':CONDITIONS[ci],'subject':sid,'site':populations()[sid]['site'],'position':pos,'original_plane_index':k,'split':'train','namespace':ns,'purpose':purpose,'selection_key':selection,'noise_key':key('center'),'neighbor_noise_keys':nkeys,'rotation':int(ar.integers(4)) if augment else 0,'flip0':bool(ar.integers(2)) if augment else False,'flip1':bool(ar.integers(2)) if augment else False}
 d=dict(zip(('x','y','y_minus','y_plus','sigma_true'),[transform(a,m) for a in (x,y,*obs,sigma)]))
 family=(index+ci)%4;nuisance_key=named_seed(ns,seed,index,CONDITIONS[ci],'nuisance_severity',purpose);nr=np.random.default_rng(nuisance_key);valid=[(r,c) for r in range(-2,3) for c in range(-2,3) if (r,c)!=(0,0)]
 shifts=[list(valid[int(nr.integers(24))]) for _ in range(2)] if family in (0,2) else [[0,0],[0,0]]
 scales=[float(np.exp(nr.uniform(np.log(.5),np.log(1.75)))) for _ in range(2)] if family in (1,2) else [1.,1.]
 dropout=('minus','plus','both')[int(nr.integers(3))] if family==3 else 'none'
 altered=[]
 for j,(a,nk,sc,dis) in enumerate(zip(pair,nkeys,scales,shifts)):
  v=obs[j] if sc==1 else rician(a,sigma*sc,np.random.default_rng(nk))
  v=shift(transform(v,m),dis)
  if dropout in (('minus' if j==0 else 'plus'),'both'):v=d['y'].copy()
  altered.append(v)
 cf={**d,'y_minus':altered[0],'y_plus':altered[1]}
 # In-plane array axes correspond to canonical axes1,0; rotation exchanges them.
 affine=np.asarray(metadata(sid)['canonical_affine'],dtype=float)[:3,:3];spacing=[float(np.linalg.norm(affine[:,1])),float(np.linalg.norm(affine[:,0]))]
 if m['rotation']%2:spacing.reverse()
 m.update(input_sha256=ah(d['y']),reference_sha256=ah(d['x']),true_sigma_sha256=ah(d['sigma_true']),neighbor_input_hashes=[ah(d[z]) for z in ('y_minus','y_plus')],counterfactual_neighbor_hashes=[ah(cf[z]) for z in ('y_minus','y_plus')],shape=list(x.shape),transformed_shape=list(d['x'].shape),family=METHOD_FAMILIES[family],nuisance_key=nuisance_key,shifts=shifts,multipliers=scales,dropout=dropout,displacements_mm=[float(np.hypot(r*spacing[0],c*spacing[1])) for r,c in shifts],display_spacing_mm=spacing,paired_center_identical=ah(cf['y'])==ah(d['y']),paired_reference_identical=ah(cf['x'])==ah(d['x']))
 assert m['paired_center_identical'] and m['paired_reference_identical']
 return d,cf,m

def stress(d,m,condition):
 if condition=='BASELINE':return d,m
 if condition in ('SHIFT1','SHIFT2'):
  s=int(condition[-1]);dd={**d,'y_minus':shift(d['y_minus'],(s,0)),'y_plus':shift(d['y_plus'],(-s,0))}
 elif condition in ('NOISE_HALF','NOISE_HIGH'):dd,_=historical_context(d,m,m['presentation_key'],.5 if condition=='NOISE_HALF' else 1.5)
 else:
  assert condition in ('DIAGNOSTIC_SHIFT','DIAGNOSTIC_NOISE')
  sid=m['subject'];pos=m['position'];k=metadata(sid)['slice_indices'][pos]
  keys=[named_seed('research17_context_robustness_diagnostic_stress_v1',m['seed'],sid,k,j,m['condition'],condition) for j in (-1,1)]
  scales=[.75,2.] if condition=='DIAGNOSTIC_NOISE' else [1.,1.]
  obs=[rician(a,d['sigma_true']*sc,np.random.default_rng(key)) for a,sc,key in zip(neighbors(sid)[pos],scales,keys)]
  if condition=='DIAGNOSTIC_SHIFT':obs=[shift(obs[0],(-3,1)),shift(obs[1],(3,-1))]
  dd={**d,'y_minus':obs[0],'y_plus':obs[1]}
  m={**m,'diagnostic_neighbor_noise_keys':keys,'diagnostic_namespace':'research17_context_robustness_diagnostic_stress_v1'}
 assert ah(dd['y'])==m['input_sha256'] and ah(dd['x'])==m['reference_sha256']
 return dd,{**m,'perturbation':condition,'stressed_neighbor_hashes':[ah(dd['y_minus']),ah(dd['y_plus'])]}

def historical_context(d,m,presentation,neighbor_scale=1.):
 sid=m['subject'];pos=m['position'];split=populations()[sid]['split'];k=metadata(sid)['slice_indices'][pos];keys=[named_seed('research17_true_adjacent_noise_v1',m['seed'],split,sid,k,j,m['condition'],presentation) for j in (k-1,k+1)]
 sig=inverse(d['sigma_true'],m)*neighbor_scale;obs=[transform(rician(a,sig,np.random.default_rng(key)),m) for a,key in zip(neighbors(sid)[pos],keys)]
 return {**d,'y_minus':obs[0],'y_plus':obs[1]},{**m,'neighbor_noise_keys':keys,'neighbor_input_hashes':[ah(a) for a in obs],'context_namespace':'research17_true_adjacent_noise_v1','presentation_key':presentation}
