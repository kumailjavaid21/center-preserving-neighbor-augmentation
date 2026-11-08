"""Matched FINAL2400 continuation with complete native-slice objectives."""
from pathlib import Path
import copy,json,os,random,time
import numpy as np
import torch
from src.io import digest,read,save,file_sha
from src.models.loading import generator,noise,expected_parent,policy
from src.restoration.preprocessing import PreparedData
from src.restoration.native import native,_plane,check_noise
from src.restoration.contextual import native as center_native
from src.augmentation import neighbor_augmentation as samples
from src.objective import optimizer,component

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])
def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value
def atomic_checkpoint(path,value):
    pending=path.with_name(path.name+'.pending')
    with pending.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(pending,path)

def train(args):
    config=read(args.config);assert config['updates_per_fit']==2400 and args.stream in config['seeds']
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the released native training path')
    policy();data=PreparedData(args.prepared_data);samples.configure(data)
    parent_spec=expected_parent(args.stream,'NAF3_SCRATCH');center_spec=expected_parent(args.stream,'CENTER')
    g=generator(args.parent_checkpoint,parent_spec['artifact_sha256'],True);n=noise(args.noise_checkpoint)
    f0=generator(args.parent_checkpoint,parent_spec['artifact_sha256'],False)
    center=generator(args.center_checkpoint,center_spec['artifact_sha256'],False)
    opt,scheduler=optimizer(g);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    latest=out/'latest.pt';started=out/'started_update.json';point=0
    binding={'config_sha256':file_sha(args.config),'stream':args.stream,'parent_sha256':parent_spec['artifact_sha256'],
             'center_sha256':center_spec['artifact_sha256'],'N_sha256':file_sha(args.noise_checkpoint)}
    if latest.with_name(latest.name+'.pending').exists():raise RuntimeError('Pending recovery requires explicit verification; preserved without overwrite')
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);assert saved['binding']==binding
        point=saved['step'];g.load_state_dict(saved['generator']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
        if started.exists() and read(started)['step']>point:raise RuntimeError('Ambiguous started update without durable advanced state')
    elif started.exists():raise RuntimeError('Started update without recovery checkpoint; no automatic repeat')
    anchors=Path(args.anchor_cache)/str(args.stream);anchors.mkdir(parents=True,exist_ok=True)
    for index in range(point,2400):
        cases=[samples.training(args.stream,index,ci) for ci in range(5)]
        anchor_path=anchors/f'{index:04d}.json';case_hash=digest([m for _,_,m in cases])
        if anchor_path.exists():
            cached=read(anchor_path);assert cached['sample_hash']==case_hash and cached['parent_sha256']==binding['parent_sha256'] and cached['center_sha256']==binding['center_sha256'] and cached['N_sha256']==binding['N_sha256']
            anchor_rows=cached['rows']
        else:
            anchor_rows=[]
            with torch.no_grad():
                for q0,_,m in cases:
                    p=native(f0,n,q0['y'],q0['y_minus'],q0['y_plus'])[0]
                    b=center_native(center,n,q0['y'],q0['y_minus'],q0['y_plus'],'CENTER')[0]
                    x=_plane(q0['x'],'cuda');anchor_rows.append({'sample_hash':digest(m),'p_mse':float((p-x).square().mean()),'b_mse':float((b-x).square().mean())})
            save(anchor_path,{'sample_hash':case_hash,'parent_sha256':binding['parent_sha256'],'center_sha256':binding['center_sha256'],'N_sha256':binding['N_sha256'],'rows':anchor_rows})
        save(started,{'step':index+1,'sample_hash':case_hash,'binding':binding})
        opt.zero_grad(set_to_none=True);values=[];check_noise(n)
        for ci,((q0,q1,m),anchor) in enumerate(zip(cases,anchor_rows)):
            assert anchor['sample_hash']==digest(m)
            x=_plane(q0['x'],'cuda');p=torch.tensor(anchor['p_mse'],device='cuda');b=torch.tensor(anchor['b_mse'],device='cuda')
            for vi,q in enumerate((q0,q0 if config['arm']=='NOMINAL' else q1)):
                prediction=native(g,n,q['y'],q['y_minus'],q['y_plus'])[0]
                mse=(prediction-x).square().mean();loss,_,_=component(mse,ci,config['arm'],vi,p,b)
                values.append(float(loss.detach()));loss.backward();del prediction,mse,loss
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in g.parameters())
        opt.step();scheduler.step()
        atomic_checkpoint(latest,{'step':index+1,'arm':config['arm'],'stream':args.stream,'binding':binding,
           'generator':cpu(g.state_dict()),'noise_estimator':cpu(n.state_dict()),'optimizer':cpu(opt.state_dict()),
           'scheduler':cpu(scheduler.state_dict()),'rng':rng_state(),'loss':sum(values),'sample_hash':case_hash})
        if (index+1)%50==0:print(config['arm'],args.stream,index+1,flush=True)
    atomic_checkpoint(out/'FINAL2400.pt',{'step':2400,'arm':config['arm'],'stream':args.stream,'binding':binding,
       'generator':cpu(g.state_dict()),'noise_estimator':cpu(n.state_dict())})
