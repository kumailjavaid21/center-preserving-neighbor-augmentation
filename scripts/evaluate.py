"""Explicit development evaluation of supplied matched-arm or frozen-parent weights."""
from pathlib import Path
import argparse,csv,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared-data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--noise-checkpoint',required=True);p.add_argument('--stream',type=int,choices=(20261117,20261118),required=True)
    p.add_argument('--method',choices=('NOMINAL','AUGMENTED','ANCHORED','NAF3_SCRATCH'),required=True);p.add_argument('--output',required=True);args=p.parse_args()
    import torch
    import numpy as np
    from src.io import ROOT,read
    from src.models.loading import generator,noise,policy,expected_parent
    from src.restoration.preprocessing import PreparedData
    from src.restoration.native import native
    from src.augmentation import neighbor_augmentation as samples
    from src.augmentation.randomness import named_seed
    from src.augmentation.rician_corruption import sigma_field,rician
    from src.augmentation.transforms import ah
    from src.evaluation.scoring import scores
    policy();source=PreparedData(args.prepared_data);samples.configure(source)
    spec=read(ROOT/'metadata/model_specifications.json')
    identity=expected_parent(args.stream,'NAF3_SCRATCH') if args.method=='NAF3_SCRATCH' else next(x for x in spec['matched_final_artifacts'] if x['seed']==args.stream and x['arm']==args.method)
    g=generator(args.checkpoint,identity['artifact_sha256']);n=noise(args.noise_checkpoint)
    collected=[]
    for sid in sorted(s for s,x in source.populations.items() if x['split']=='validation'):
        for pos in (0,4,7,11):
            for ci,condition in enumerate(samples.CONDITIONS):
                x=source.reference(sid,pos);rng=np.random.default_rng(named_seed('validation_corruption',sid,pos,condition))
                sigma=sigma_field(x.shape,rng,condition);y=rician(x,sigma,rng)
                meta={'seed':args.stream,'subject':sid,'position':pos,'condition':condition,'condition_index':ci,'site':source.populations[sid]['site'],
                  'rotation':0,'flip0':False,'flip1':False,'input_sha256':ah(y),'reference_sha256':ah(x),'true_sigma_sha256':ah(sigma)}
                d,meta=samples.historical_context({'x':x,'y':y,'sigma_true':sigma},meta,f'validation/{pos}/{condition}')
                for stress in (('BASELINE','SHIFT1','SHIFT2','NOISE_HALF','NOISE_HIGH','DIAGNOSTIC_SHIFT','DIAGNOSTIC_NOISE') if pos==7 else ('BASELINE',)):
                    data,m=samples.stress(d,dict(meta),stress)
                    with torch.no_grad():prediction=native(g,n,data['y'],data['y_minus'],data['y_plus'])[0].cpu().numpy()
                    collected.extend(scores(prediction,x,{**m,'method':args.method,'perturbation':stress}))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='') as f:
        keys=list(dict.fromkeys(k for row in collected for k in row));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(collected)
if __name__=='__main__':main()
