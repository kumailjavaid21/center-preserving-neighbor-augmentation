"""CPU-only reconstruction of the existing subject-first scalar tables."""
import collections,csv,math
from pathlib import Path

METRICS=('psnr','ssim','foreground_psnr','edge_mae','mae','mse_raw','foreground_mse_raw')
CONDITIONS=('uniform_003','uniform_006','uniform_009','spatial','clean')
PRIMARY=('SHIFT1','SHIFT2','NOISE_HIGH')
METHODS=('ANCHORED','AUGMENTED','CENTER','COMPAT3','JIANG','JIANG_BNCAL','NAF3_SCRATCH','NOMINAL','PLAIN3')
def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def average(rows):return {m:math.fsum(float(x[m]) for x in rows)/len(rows) for m in METRICS}
def subject_rows(rows,positions,stresses=None):
    if stresses is not None:rows=[x for x in rows if x['perturbation'] in stresses]
    groups=collections.defaultdict(list);seen=set()
    for row in rows:
        key=tuple(row[k] for k in ('seed','method','policy','subject','site','condition'))
        unique=key+(row['position'],row.get('perturbation','STANDARD'))
        if unique in seen:raise ValueError('Duplicate case')
        seen.add(unique);groups[key].append(row)
        if any(not math.isfinite(float(row[m])) for m in METRICS):raise ValueError('Nonfinite scalar')
    subjects=[]
    for key,rr in sorted(groups.items()):
        if {int(x['position']) for x in rr}!=set(positions):raise ValueError('Incomplete position set')
        if len(rr)!=len(positions)*(len(stresses) if stresses else 1):raise ValueError('Incomplete case grid')
        subjects.append({**dict(zip(('seed','method','policy','subject','site','condition'),key)),**average(rr)})
    noisy=collections.defaultdict(list)
    for row in subjects:
        if row['condition']!='clean':noisy[tuple(row[k] for k in ('seed','method','policy','subject','site'))].append(row)
    for key,rr in sorted(noisy.items()):
        if len(rr)!=4 or {x['condition'] for x in rr}!=set(CONDITIONS[:-1]):raise ValueError('Noisy condition grid incomplete')
        subjects.append({**dict(zip(('seed','method','policy','subject','site'),key)),'condition':'noisy_mean',**average(rr)})
        high=[x for x in rr if x['condition'] in ('uniform_009','spatial')]
        if len(high)!=2:raise ValueError('High/spatial condition grid incomplete')
        subjects.append({**dict(zip(('seed','method','policy','subject','site'),key)),'condition':'high_spatial',**average(high)})
    pooled=collections.defaultdict(list)
    for row in subjects:pooled[tuple(row[k] for k in ('method','policy','subject','site','condition'))].append(row)
    for key,rr in sorted(pooled.items()):
        if {x['seed'] for x in rr}!={'20261117','20261118'} or len(rr)!=2:raise ValueError('Must pool two streams within the same subject')
        subjects.append({**dict(zip(('method','policy','subject','site','condition'),key)),'seed':'pooled_within_subject',**average(rr)})
    return subjects
def absolute(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in ('seed','method','policy','condition'))].append(row)
    out=[]
    for key,rr in sorted(groups.items()):
        if len(rr)!=16 or len({x['subject'] for x in rr})!=16:raise ValueError('Subject count must remain 16')
        out.append({**dict(zip(('seed','method','policy','condition'),key)),'subjects':16,**average(rr)})
    return out
def all_subjects(root):
    root=Path(root);std=read_csv(root/'results/standard_cases.csv');stress=read_csv(root/'results/stress_cases.csv')
    if len(std)!=11520 or len(stress)!=20160:raise ValueError('Unexpected complete-development case counts')
    results={'STANDARD':subject_rows(std,(0,4,7,11)), 'PRIMARY_THREE_STRESS':subject_rows(stress,(7,),PRIMARY)}
    for name in ('BASELINE',*PRIMARY,'NOISE_HALF','DIAGNOSTIC_SHIFT','DIAGNOSTIC_NOISE'):
        results['STRESS/'+name]=subject_rows([x for x in stress if x['perturbation']==name],(7,))
    return results
def verify(root,tolerance=1e-10):
    root=Path(root);sets=all_subjects(root);checked=0
    indexes={name:{tuple(x[k] for k in ('seed','method','policy','subject','condition')):x for x in rr} for name,rr in sets.items()}
    absolutes={name:{tuple(x[k] for k in ('seed','method','policy','condition')):x for x in absolute(rr)} for name,rr in sets.items()}
    for name,file in [('STANDARD','standard_metrics.csv'),('PRIMARY_THREE_STRESS','robustness_metrics.csv')]:
        recomputed=absolutes[name]
        published=read_csv(root/'results'/file)
        if len(published)!=len(recomputed):raise ValueError('Absolute table row count differs')
        for row in published:
            actual=recomputed[tuple(row[k] for k in ('seed','method','policy','condition'))]
            for m in METRICS:
                if abs(float(row[m])-actual[m])>tolerance:raise ValueError('Aggregate differs: '+file+' '+m)
            checked+=1
    for row in read_csv(root/'results/stress_scenarios.csv'):
        name='STRESS/'+row['perturbation'];recomputed=absolutes[name]
        actual=recomputed[tuple(row[k] for k in ('seed','method','policy','condition'))]
        for m in METRICS:
            if abs(float(row[m])-actual[m])>tolerance:raise ValueError('Stress aggregate differs')
        checked+=1
    for row in read_csv(root/'results/subject_level_effects.csv'):
        name=row['scenario']+('/'+row['perturbation'] if row['scenario']=='STRESS' else '')
        a,b=row['comparison'].split('_minus_');common=(row['seed'],row['policy'],row['subject'],row['condition'])
        ra=indexes[name][(common[0],a,*common[1:])];rb=indexes[name][(common[0],b,*common[1:])]
        for m in METRICS:
            if abs((ra[m]-rb[m])-float(row[m]))>tolerance:raise ValueError('Paired subject effect differs')
        checked+=1
    for row in read_csv(root/'results/ablation_comparisons.csv'):
        name=row['scenario']+('/'+row['perturbation'] if row['scenario']=='STRESS' else '')
        a,b=row['comparison'].split('_minus_');metric=row['metric'];idx=indexes[name]
        ids=sorted({x['subject'] for x in sets[name]})
        values=[idx[(row['seed'],a,row['policy'],sid,row['condition'])][metric]-idx[(row['seed'],b,row['policy'],sid,row['condition'])][metric] for sid in ids]
        if abs(math.fsum(values)/16-float(row['mean']))>tolerance:raise ValueError('Contrast cohort mean differs')
        if float(row['CI95_low'])>float(row['CI95_high']):raise ValueError('Invalid original interval ordering')
        checked+=1
    return {'status':'passed','scalar_rows_checked':checked,'raw_case_rows':31680,'subjects':16,'streams_pooled_within_subject':2,'tolerance':tolerance,'new_model_evaluations':0}
