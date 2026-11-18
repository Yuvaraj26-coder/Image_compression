import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from svdlab.decompose import compute_svd, reconstruct, truncate_svd
from svdlab.metrics import (
    frobenius_error,
    mse,
    psnr,
    spectral_error,
    ssim,
)


def test_uint8_mse_no_underflow():
    a = np.array([[0]], dtype=np.uint8)
    b = np.array([[255]], dtype=np.uint8)
    assert mse(a, b) == 65025.0


def test_psnr_uint8_no_underflow():
    a = np.array([[0, 255]], dtype=np.uint8)
    b = np.array([[255, 0]], dtype=np.uint8)
    assert psnr(a, b) == pytest.approx(
        20 * np.log10(255.0) - 10 * np.log10(mse(a, b)), abs=1e-6
    )


def test_psnr_identical_is_infinite():
    a = np.ones((4, 4))
    assert np.isinf(psnr(a, a))


@pytest.mark.parametrize("shape", [(20, 15), (12, 12)])
def test_psnr_matches_skimage(shape):
    rng = np.random.default_rng(42)
    a = rng.integers(0, 256, size=shape)
    b = np.clip(a + rng.integers(-5, 6, size=shape), 0, 255)
    assert psnr(a, b) == pytest.approx(
        peak_signal_noise_ratio(a, b, data_range=255), abs=1e-6
    )


def test_ssim_real_number():
    a = np.random.default_rng(1).integers(0, 256, (20, 20))
    assert -1 <= ssim(a, a.copy()) <= 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_frobenius_error_matches_measurement(k):
    rng = np.random.default_rng(7)
    A = rng.normal(size=(8, 6))
    U, S, Vt = compute_svd(A)
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, k)
    Ak = reconstruct(Uk, Sk, Vtk)
    assert frobenius_error(S, k) == pytest.approx(np.linalg.norm(A - Ak), abs=1e-8)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_spectral_error_matches_measurement(k):
    rng = np.random.default_rng(8)
    A = rng.normal(size=(8, 6))
    U, S, Vt = compute_svd(A)
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, k)
    Ak = reconstruct(Uk, Sk, Vtk)
    assert spectral_error(S, k) == pytest.approx(
        np.linalg.norm(A - Ak, ord=2), abs=1e-8
    )
