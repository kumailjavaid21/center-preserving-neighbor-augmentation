"""Load explicitly supplied, separately obtained parent/final checkpoint files."""
from pathlib import Path
import torch
from .generator import Generator
from .noise_estimator import NoiseEstimator
from src.io import ROOT,read,file_sha

def policy():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)

def expected_parent(stream,method):
    return next(x for x in read(ROOT/'metadata/model_specifications.json')['required_initial_parent_artifacts'] if x['stream']==stream and x['method']==method)

def generator(path,expected_sha,train=False):
    if file_sha(path)!=expected_sha:raise ValueError('Checkpoint bytes do not match the declared artifact')
    saved=torch.load(Path(path),map_location='cpu',weights_only=False)
    g=Generator().cuda();g.method='NAF3_SCRATCH';g.load_state_dict(saved['generator']);g.train(train);g.requires_grad_(train)
    return g

def noise(path):
    spec=read(ROOT/'metadata/model_specifications.json')['noise_estimator']
    if file_sha(path)!=spec['artifact_sha256']:raise ValueError('Frozen N artifact mismatch')
    saved=torch.load(Path(path),map_location='cpu',weights_only=False)
    n=NoiseEstimator('contrast').cuda();n.load_state_dict(saved['noise_estimator']);return n.freeze()
