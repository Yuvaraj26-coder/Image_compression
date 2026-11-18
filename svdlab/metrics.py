from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    a = np.asarray(image1).astype(np.float64)
    b = np.asarray(image2).astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(image1: np.ndarray, image2: np.ndarray, data_range: float = 255.0) -> float:
    error = mse(image1, image2)
    if error == 0:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(error))


def ssim(image1: np.ndarray, image2: np.ndarray, data_range: float = 255.0) -> float:
    a = np.asarray(image1)
    b = np.asarray(image2)
    channel_axis = 2 if a.ndim == 3 else None
    return float(
        structural_similarity(a, b, data_range=data_range, channel_axis=channel_axis)
    )


def compression_ratio(original_size: int, compressed_size: int) -> float:
    if compressed_size <= 0:
        raise ValueError("compressed_size must be positive")
    return float(original_size) / float(compressed_size)


def frobenius_error(S: np.ndarray, k: int) -> float:
    values = np.asarray(S, dtype=np.float64)
    k = max(0, min(int(k), len(values)))
    if k >= len(values):
        return 0.0
    return float(np.sqrt(np.sum(values[k:] ** 2)))


def spectral_error(S: np.ndarray, k: int) -> float:
    values = np.asarray(S, dtype=np.float64)
    if k >= len(values) or k < 0:
        return 0.0
    return float(values[k])
