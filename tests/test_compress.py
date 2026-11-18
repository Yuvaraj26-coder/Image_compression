import numpy as np
import pytest

from svdlab.compress import (
    break_even_rank,
    dequantize_factors,
    deserialize_compressed,
    quantize_factors,
    serialize_compressed,
    theoretical_storage,
)
from svdlab.decompose import compute_svd, reconstruct, truncate_svd
from svdlab.metrics import compression_ratio, psnr


def test_quantization_round_trip():
    rng = np.random.default_rng(12)
    A = rng.uniform(0, 255, (30, 25))
    U, S, Vt = compute_svd(A)
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, 25)
    quantized = quantize_factors(Uk, Sk, Vtk)
    payload = serialize_compressed(quantized, A.shape)
    restored, shape = deserialize_compressed(payload)
    Uq, Sq, Vtq = dequantize_factors(restored)
    result = reconstruct(Uq, Sq, Vtq)
    assert shape == A.shape
    assert psnr(A, result) > 35


def test_compression_ratio():
    assert compression_ratio(100, 25) == 4


@pytest.mark.parametrize("shape", [(20, 5), (5, 20), (20, 20), (1, 1)])
def test_storage_formula(shape):
    m, n = shape
    assert theoretical_storage(m, n, 3) == 3 * (m + n + 1)
    assert break_even_rank(m, n) == m * n / (m + n + 1)


def test_quantized_values_are_int16():
    U = np.eye(3)
    S = np.ones(3)
    Vt = np.eye(3)
    q = quantize_factors(U, S, Vt)
    assert q["U"].dtype == np.int16
    assert q["Vt"].dtype == np.int16
    assert q["S"].dtype == np.float32


def test_serialized_payload_is_npz():
    q = quantize_factors(np.eye(2), np.ones(2), np.eye(2))
    payload = serialize_compressed(q, (2, 2))
    assert payload[:4] == b"PK\x03\x04"
