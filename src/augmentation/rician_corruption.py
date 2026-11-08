import numpy as np
import torch
from torch.nn import functional as F

def sigma_field(shape, rng, condition):
    if condition == 'clean':
        return np.zeros(shape, np.float32)
    if condition.startswith('uniform_'):
        level = {'uniform_003': 0.03, 'uniform_006': 0.06, 'uniform_009': 0.09}[condition]
        return np.full(shape, level, np.float32)
    if condition == 'uniform':
        return np.full(shape, rng.uniform(.01, .10), np.float32)
    grid = torch.from_numpy(rng.uniform(.01, .10, (1, 1, 4, 4)).astype(np.float32))
    return F.interpolate(grid, shape, mode='bilinear', align_corners=False)[0, 0].numpy()

def rician(reference, sigma_true, rng):
    e1 = rng.standard_normal(reference.shape, dtype=np.float32)
    e2 = rng.standard_normal(reference.shape, dtype=np.float32)
    # Never clip noisy magnitudes at one.
    return np.sqrt((reference+sigma_true*e1)**2+(sigma_true*e2)**2).astype(np.float32)
