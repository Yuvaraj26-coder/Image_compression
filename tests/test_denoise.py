import numpy as np

from svdlab.decompose import (
    compute_svd,
    denoise_matrix,
    gavish_donoho_threshold,
)


def test_gavish_threshold_formula():
    S = np.array([10.0, 9.0, 2.0, 1.0])
    threshold, omega = gavish_donoho_threshold(S, 4, 4)
    expected_omega = 0.56 - 0.95 + 1.82 + 1.43
    assert omega == expected_omega
    assert threshold == expected_omega * np.median(S)


def test_synthetic_low_rank_rank_recovery():
    rng = np.random.default_rng(123)
    left = rng.normal(size=(40, 5))
    right = rng.normal(size=(5, 40))
    clean = left @ right
    noisy = clean + rng.normal(0, 0.1, clean.shape)
    U, S, Vt = compute_svd(noisy)
    threshold, _ = gavish_donoho_threshold(S, *noisy.shape)
    denoised = denoise_matrix(noisy, threshold)
    recovered_rank = np.sum(S > threshold)
    assert 3 <= recovered_rank <= 10
    assert np.mean((denoised - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_threshold_is_positive():
    threshold, _ = gavish_donoho_threshold(np.arange(1, 11), 10, 10)
    assert threshold > 0
