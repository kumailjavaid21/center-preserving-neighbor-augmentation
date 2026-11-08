import numpy as np
import torch
from torch.nn import functional as F
from .native_tiling import geometry,blend_tiles

def check_noise(model):
    if model is None:
        return
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())

def _plane(value, device):
    if torch.is_tensor(value):
        assert value.dtype == torch.float32 and value.ndim == 2
        return value.to(device)
    assert isinstance(value, np.ndarray) and value.dtype == np.float32 and value.ndim == 2
    return torch.as_tensor(value, device=device)

def _chunks(values, microbatch):
    for start in range(0, len(values), microbatch):
        yield torch.cat(values[start:start + microbatch])

def prepare(model, n, y, y_minus, y_plus):
    """Inference features depend solely on the three supplied observations."""
    assert y.shape == y_minus.shape == y_plus.shape
    check_noise(n)
    device = next(model.parameters()).device
    layout = geometry(y.shape)
    h, w = layout['shape']
    hp, wp = layout['padded_shape']
    center, minus, plus = [_plane(value, device) for value in (y, y_minus, y_plus)]
    if model.method == 'NAF3_SCRATCH':
        minus, plus = minus - center, plus - center
    channels = []
    for value in (center, minus, plus):
        padded = F.pad(value[None, None], (0, wp - w, 0, hp - h), mode='replicate')
        channels.append([padded[:, :, r:r + 128, c:c + 128].contiguous()
                         for r, c in layout['positions']])
    sigmas = []
    if model.method == 'NAF3_SCRATCH':
        assert n is not None
        with torch.no_grad():
            for batch in _chunks(channels[0], 4):
                sigma = n(batch)
                sigmas.extend(sigma[j:j + 1].contiguous() for j in range(len(batch)))
    else:
        sigmas = [None] * layout['tiles']
    return channels, sigmas, layout

def native(model, n, y, y_minus, y_plus):
    channels, sigmas, layout = prepare(model, n, y, y_minus, y_plus)
    microbatch = model._G_microbatch
    assert microbatch == 4
    outputs, sizes = [], []
    for start in range(0, layout['tiles'], microbatch):
        args = [torch.cat(v[start:start + microbatch]) for v in channels]
        sigma = torch.cat(sigmas[start:start + microbatch]) if model.method == 'NAF3_SCRATCH' else None
        output = model(args[0], sigma, args[1], args[2])
        outputs.append(output[:, 0])
        sizes.append(len(output))
    prediction = blend_tiles(torch.cat(outputs), layout)
    native_sigma = blend_tiles(torch.cat([s[:, 0] for s in sigmas]), layout) if model.method == 'NAF3_SCRATCH' else None
    assert torch.isfinite(prediction).all()
    used_n = model.method == 'NAF3_SCRATCH'
    diagnostic = {'tiles': layout['tiles'], 'G_native_passes': 1,
                  'G_tile_evaluations': layout['tiles'], 'G_calls': len(outputs),
                  'G_microbatch': microbatch, 'G_microbatch_sizes': sizes,
                  'N_native_passes': int(used_n), 'N_tile_evaluations': layout['tiles'] * int(used_n),
                  'N_calls': len(outputs) * int(used_n), 'N_microbatch': 4 if used_n else 0,
                  'BN_training': bool(model.training and model.method == 'JIANG'),
                  'BN_forward_batches': len(outputs) if model.training and model.method == 'JIANG' else 0,
                  'complete_native_render': True, 'activation_recomputation': False}
    return prediction, native_sigma, diagnostic
