import numpy as np
import pytest

from svdlab.decompose import (
    cumulative_energy,
    rank_for_energy,
    rank_one_component,
    reconstruct,
    truncate_svd,
)


@pytest.mark.parametrize("shape", [(3, 5), (8, 8), (2, 7)])
def test_full_rank_svd_reconstruction(shape):
    rng = np.random.default_rng(100)
    matrix = rng.normal(size=shape)
    from svdlab.decompose import compute_svd

    U, S, Vt = compute_svd(matrix)
    assert np.max(np.abs(reconstruct(U, S, Vt) - matrix)) < 1e-8


@pytest.mark.parametrize("k", [1, 2, 4])
def test_truncation_shapes(k):
    U, S, Vt = __import__("svdlab.decompose", fromlist=["compute_svd"]).compute_svd(
        np.arange(20).reshape(4, 5)
    )
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, k)
    assert Uk.shape == (4, k)
    assert Sk.shape == (k,)
    assert Vtk.shape == (k, 5)


def test_rank_one_components_sum():
    matrix = np.arange(20, dtype=float).reshape(4, 5)
    U, S, Vt = __import__("svdlab.decompose", fromlist=["compute_svd"]).compute_svd(
        matrix
    )
    components = [rank_one_component(U, S, Vt, i) for i in range(3)]
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, 3)
    assert np.allclose(sum(components), reconstruct(Uk, Sk, Vtk), atol=1e-12)


def test_blank_energy_guard():
    energy = cumulative_energy(np.zeros(5))
    assert np.array_equal(energy, np.zeros(5))


@pytest.mark.parametrize("target", [0.0, 0.5, 0.95, 1.0])
def test_rank_for_energy(target):
    S = np.array([3.0, 2.0, 1.0])
    rank = rank_for_energy(S, target)
    assert rank >= 1
    assert cumulative_energy(S)[rank - 1] + 1e-12 >= target


def test_k_larger_than_rank_is_clamped():
    U, S, Vt = __import__("svdlab.decompose", fromlist=["compute_svd"]).compute_svd(
        np.eye(3)
    )
    Uk, Sk, Vtk = truncate_svd(U, S, Vt, 99)
    assert len(Sk) == 3
