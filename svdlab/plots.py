from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_image_matrix(image: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(np.clip(image, 0, 255), cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_patch_heatmap(patch: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = np.asarray(patch)
    plot = ax.imshow(image, cmap="viridis", vmin=0, vmax=255)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            ax.text(
                col,
                row,
                f"{image[row, col]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    ax.set_title(title)
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    return fig


def plot_singular_vectors(
    U: np.ndarray, Vt: np.ndarray, n_vectors: int = 5
) -> plt.Figure:
    n = min(n_vectors, U.shape[1], Vt.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].imshow(U[:, :n], aspect="auto", cmap="coolwarm")
    axes[0].set_title("Left singular vectors U")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Row")
    axes[1].imshow(Vt[:n, :], aspect="auto", cmap="coolwarm")
    axes[1].set_title("Right singular vectors Vᵀ")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Component")
    fig.tight_layout()
    return fig


def plot_rank_one_components(
    components: list[np.ndarray], n_components: int = 6
) -> plt.Figure:
    components = components[:n_components]
    cols = min(3, len(components))
    rows = int(np.ceil(len(components) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)
    limit = max((float(np.max(np.abs(a))) for a in components), default=1.0)
    for i, component in enumerate(components):
        ax = axes.flat[i]
        ax.imshow(component, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"Rank-1 component {i + 1}")
        ax.axis("off")
    for ax in axes.flat[len(components) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_scree(S: np.ndarray, k: int | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.asarray(S), marker=".")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("Scree plot")
    if k is not None:
        ax.axvline(k - 1, color="red", linestyle="--", label=f"k = {k}")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_energy(
    S: np.ndarray, k: int | None = None, targets: list[float] | None = None
) -> plt.Figure:
    values = np.asarray(S, dtype=np.float64)
    total = np.sum(values * values)
    energy = np.cumsum(values * values) / total if total > 0 else np.zeros_like(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(100 * energy, marker=".")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_ylim(0, 101)
    ax.grid(True, alpha=0.3)
    for target in targets or [0.90, 0.95, 0.99]:
        rank = int(np.searchsorted(energy, target, side="left") + 1)
        ax.axvline(rank - 1, linestyle="--", alpha=0.7)
        ax.text(
            rank - 1, target * 100 - 3, f"{int(target * 100)}%: k={rank}", fontsize=8
        )
    if k is not None:
        ax.axvline(k - 1, color="red", label=f"k = {k}")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray, reconstructed: np.ndarray
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(
        np.clip(original, 0, 255), cmap="gray" if original.ndim == 2 else None
    )
    axes[0].set_title("Original")
    axes[1].imshow(
        np.clip(reconstructed, 0, 255), cmap="gray" if original.ndim == 2 else None
    )
    axes[1].set_title("Reconstruction")
    error = np.abs(np.asarray(original, dtype=np.float64) - reconstructed)
    axes[2].imshow(error, cmap="inferno", vmin=0, vmax=max(float(np.max(error)), 1))
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_filmstrip(images: list[np.ndarray], k_values: list[int]) -> plt.Figure:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3), squeeze=False)
    for i, (image, k) in enumerate(zip(images, k_values, strict=False)):
        axes[0, i].imshow(
            np.clip(image, 0, 255), cmap="gray" if image.ndim == 2 else None
        )
        axes[0, i].set_title(f"k = {k}")
        axes[0, i].axis("off")
    fig.tight_layout()
    return fig


def plot_error_heatmap(original: np.ndarray, reconstructed: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    error = np.abs(np.asarray(original, dtype=np.float64) - reconstructed)
    vmax = max(float(np.max(error)), 1.0)
    plot = ax.imshow(error, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("Absolute error heatmap")
    ax.axis("off")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    return fig


def plot_psnr_ssim_vs_rank(
    ranks: list[int],
    psnrs: list[float],
    ssims: list[float],
    current_k: int | None = None,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(ranks, psnrs, color="tab:blue", marker=".", label="PSNR")
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ranks, ssims, color="tab:orange", marker=".", label="SSIM")
    ax2.set_ylabel("SSIM", color="tab:orange")
    if current_k is not None:
        ax1.axvline(current_k, color="red", linestyle="--")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rate_distortion(
    svd_sizes: list[float],
    svd_psnrs: list[float],
    jpeg_sizes: list[float],
    jpeg_psnrs: list[float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(svd_sizes, svd_psnrs, marker=".", label="SVD")
    ax.plot(jpeg_sizes, jpeg_psnrs, marker=".", label="JPEG")
    ax.set_xlabel("File size (bytes)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-distortion comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_denoise_comparison(
    noisy: np.ndarray, denoised: np.ndarray, residual: np.ndarray
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in zip(
        axes,
        [noisy, denoised, residual],
        ["Noisy", "Denoised", "Residual"],
        strict=False,
    ):
        if title == "Residual":
            vmax = max(float(np.max(np.abs(residual))), 1.0)
            ax.imshow(residual, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(np.clip(image, 0, 255), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_spectrum_threshold(S: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.asarray(S), marker=".")
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Gavish-Donoho τ = {threshold:.4f}",
    )
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.set_title("Thresholded singular spectrum")
    ax.legend()
    fig.tight_layout()
    return fig
