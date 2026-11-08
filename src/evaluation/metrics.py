import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

def image_metrics(pred, ref):
    pred, ref = np.asarray(pred, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    error = pred-ref
    mse = float(np.mean(error**2))
    mask = ref > 0.05  # Fixed reference-derived foreground mask.
    fmse = float(np.mean(error[mask]**2)) if mask.any() else mse
    # scipy Sobel (reflect boundary), divide by 8 in 2D; compare both components.
    edge = np.mean([np.mean(np.abs(ndimage.sobel(pred, axis=k, mode='reflect')/8 -
                                   ndimage.sobel(ref, axis=k, mode='reflect')/8)) for k in (0, 1)])
    return {'mse_raw': mse, 'psnr': float(-10*np.log10(mse)) if mse > 0 else float('inf'),
            'ssim': float(structural_similarity(ref, pred, data_range=1, win_size=11,
                                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
            'foreground_mse_raw': fmse, 'foreground_psnr': float(-10*np.log10(fmse)) if fmse > 0 else float('inf'),
            'edge_mae': float(edge), 'mae': float(np.mean(np.abs(error)))}
