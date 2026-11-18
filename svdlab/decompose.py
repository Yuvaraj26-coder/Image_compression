from __future__ import annotations

import numpy as np


def compute_svd(
    matrix: np.ndarray, full_matrices: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("SVD input must be a 2D array")
    U, S, Vt = np.linalg.svd(array, full_matrices=full_matrices)
    return U, S, Vt


def truncate_svd(
    U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(max(1, min(k, len(S))))
    return U[:, :k], S[:k], Vt[:k, :]


def rank_one_component(
    U: np.ndarray, S: np.ndarray, Vt: np.ndarray, i: int
) -> np.ndarray:
    if i < 0 or i >= len(S):
        raise IndexError("Singular component index out of range")
    return np.asarray(S[i] * np.outer(U[:, i], Vt[i, :]), dtype=np.float64)


def reconstruct(U_k: np.ndarray, S_k: np.ndarray, Vt_k: np.ndarray) -> np.ndarray:
    return np.asarray((U_k * S_k) @ Vt_k, dtype=np.float64)


def cumulative_energy(S: np.ndarray) -> np.ndarray:
    values = np.asarray(S, dtype=np.float64)
    total = np.sum(values * values)
    if total <= 0:
        return np.zeros_like(values)
    return np.cumsum(values * values) / total


def rank_for_energy(S: np.ndarray, target_energy: float = 0.95) -> int:
    target = float(min(1.0, max(0.0, target_energy)))
    energy = cumulative_energy(S)
    if len(energy) == 0:
        return 1
    return int(np.searchsorted(energy, target, side="left") + 1)


def gavish_donoho_threshold(S: np.ndarray, m: int, n: int) -> tuple[float, float]:
    values = np.asarray(S, dtype=np.float64)
    if m <= 0 or n <= 0:
        raise ValueError("Matrix dimensions must be positive")
    beta = min(m, n) / float(max(m, n))
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median = float(np.median(values)) if len(values) else 0.0
    threshold = omega * median
    return threshold, omega


def denoise_matrix(matrix: np.ndarray, threshold: float) -> np.ndarray:
    U, S, Vt = compute_svd(matrix)
    kept = np.maximum(S - threshold, 0.0)
    return reconstruct(U, kept, Vt)
