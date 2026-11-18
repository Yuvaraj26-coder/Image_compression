from __future__ import annotations

import hashlib
import io
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from svdlab.compress import (
    break_even_rank,
    dequantize_factors,
    quantize_factors,
    serialize_compressed,
    theoretical_storage,
)
from svdlab.decompose import (
    compute_svd,
    denoise_matrix,
    gavish_donoho_threshold,
    rank_for_energy,
    rank_one_component,
    reconstruct,
)
from svdlab.io import load_image
from svdlab.metrics import (
    compression_ratio,
    frobenius_error,
    psnr,
    spectral_error,
    ssim,
)
from svdlab.plots import (
    plot_cumulative_energy,
    plot_denoise_comparison,
    plot_filmstrip,
    plot_image_matrix,
    plot_patch_heatmap,
    plot_psnr_ssim_vs_rank,
    plot_rank_one_components,
    plot_rate_distortion,
    plot_reconstruction_comparison,
    plot_scree,
    plot_singular_vectors,
    plot_spectrum_threshold,
)


@st.cache_data(show_spinner=False)
def cached_load(
    source: Any, sample_name: str, max_side: int
) -> tuple[np.ndarray, dict]:
    return load_image(source, sample_name, max_side)


@st.cache_data(show_spinner=False)
def compute_channel_svds(
    channel_images: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [compute_svd(channel) for channel in channel_images]


@st.cache_data(show_spinner=False)
def compute_denoising(
    channel_images: list[np.ndarray], noise_sigma: float, mode: str
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    noisy_channels: list[np.ndarray] = []
    denoised_channels: list[np.ndarray] = []
    residual_channels: list[np.ndarray] = []
    thresholds: list[float] = []
    rng = np.random.default_rng(2026)
    for channel in channel_images:
        noisy = np.clip(channel + rng.normal(0, noise_sigma, channel.shape), 0, 255)
        U, S, Vt = compute_svd(noisy)
        threshold, _ = gavish_donoho_threshold(S, noisy.shape[0], noisy.shape[1])
        denoised = denoise_matrix(noisy, threshold)
        denoised = np.clip(denoised, 0, 255)
        noisy_channels.append(noisy)
        denoised_channels.append(denoised)
        residual_channels.append(noisy - denoised)
        thresholds.append(threshold)
    return noisy_channels, denoised_channels, residual_channels, thresholds


def render_plot(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


def channel_reconstruct(
    svds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], k: int
) -> np.ndarray:
    reconstructed = [
        reconstruct(
            *[
                part[:k] if i == 1 else part[:, :k] if i == 0 else part[:k, :]
                for i, part in enumerate(svd)
            ]
        )
        for svd in svds
    ]
    if len(reconstructed) == 1:
        return np.clip(reconstructed[0], 0, 255)
    stacked = np.stack(reconstructed, axis=2)
    return np.clip(stacked, 0, 255)


def channel_slice(svd: tuple[np.ndarray, np.ndarray, np.ndarray], k: int) -> tuple:
    U, S, Vt = svd
    return U[:, :k], S[:k], Vt[:k, :]


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def quality_analysis(
    channel_svds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    channels: list[np.ndarray],
) -> tuple[
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    max_rank = min([len(svd[1]) for svd in channel_svds])
    ranks = sorted(
        set(
            [1, 2, 5, 10, 20, 50, 100]
            + list(np.unique(np.geomspace(1, max_rank, 40).astype(int)))
        )
    )
    ranks = [k for k in ranks if k <= max_rank]
    psnrs: list[float] = []
    ssims: list[float] = []
    svd_sizes: list[float] = []
    jpeg_sizes: list[float] = []
    jpeg_psnrs: list[float] = []
    for k in ranks:
        reconstructed = channel_reconstruct(channel_svds, k)
        original = np.stack(channels, axis=2) if len(channels) > 1 else channels[0]
        psnrs.append(psnr(original, reconstructed))
        ssims.append(ssim(original, reconstructed))
        m, n = channels[0].shape
        svd_sizes.append((k * (m + n + 1) * len(channels) * 4))
        quality = int(np.clip(95 - k / max(max_rank, 1) * 70, 5, 95))
        rgb = np.clip(reconstructed, 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb if rgb.ndim == 3 else rgb)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality, optimize=True)
        jpeg_sizes.append(len(buffer.getvalue()))
        jpeg_rgb = np.asarray(
            Image.open(io.BytesIO(buffer.getvalue())).convert(
                "RGB" if rgb.ndim == 3 else "L"
            ),
            dtype=np.float64,
        )
        jpeg_psnrs.append(psnr(original, jpeg_rgb))
        jpeg_psnrs[-1] = jpeg_psnrs[-1] if np.isfinite(jpeg_psnrs[-1]) else 0.0
    return ranks, psnrs, ssims, svd_sizes, psnrs, jpeg_sizes, jpeg_psnrs


def main() -> None:
    st.set_page_config(page_title="SVD Image Compressor", layout="wide")
    st.title("SVD Image Compressor")
    st.caption("An Interactive Walkthrough of Low-Rank Image Compression")

    with st.sidebar:
        st.header("Image source")
        source_choice = st.radio("Source", ["Sample", "Upload"], horizontal=True)
        sample_name = st.selectbox("Sample", ["camera", "astronaut", "coffee"])
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
        max_side = st.slider("Maximum side length", 64, 512, 256, 16)
        color_mode = st.selectbox(
            "Color processing",
            ["Grayscale", "Color (RGB per channel)", "Color (YCbCr)"],
        )

    if source_choice == "Upload" and uploaded is not None:
        source: Any = uploaded
    else:
        source = "sample"
    image, info = cached_load(source, sample_name, max_side)

    if color_mode == "Grayscale":
        if image.ndim == 3:
            from skimage.color import rgb2gray

            display_image = np.clip(rgb2gray(image) * 255, 0, 255)
            channels = [display_image]
        else:
            display_image = image
            channels = [image]
    elif color_mode == "Color (YCbCr)":
        from skimage.color import rgb2ycbcr

        if image.ndim == 2:
            display_image = np.repeat(image[:, :, None], 3, axis=2)
        else:
            display_image = np.clip(image, 0, 255)
        ycbcr = rgb2ycbcr(np.clip(display_image, 0, 255) / 255)
        channels = [ycbcr[:, :, i] * (255 if i == 0 else 255) for i in range(3)]
    else:
        display_image = np.clip(image, 0, 255)
        if display_image.ndim == 2:
            channels = [display_image]
        else:
            channels = [display_image[:, :, i] for i in range(3)]

    original = np.stack(channels, axis=2) if len(channels) > 1 else channels[0]
    svds = compute_channel_svds(channels)

    stage = st.radio(
        "Guided stages",
        [
            "1. Image as a Matrix",
            "2. The Decomposition",
            "3. Where the Information Lives",
            "4. Building the Image Up",
            "5. The Compression Maths",
            "6. Quality Analysis",
            "Bonus: Image Denoising",
        ],
        horizontal=True,
    )

    if stage.startswith("1."):
        render_plot(plot_image_matrix(display_image, "Loaded image"))
        st.write(
            {
                "shape": original.shape,
                "dtype": str(original.dtype),
                "elements": int(original.size),
                "raw_bytes": int(original.nbytes),
                "source": info,
            }
        )
        h, w = channels[0].shape
        ph, pw = min(8, h), min(8, w)
        patch = channels[0][:ph, :pw]
        st.subheader("Interactive 8×8 patch")
        st.dataframe(pd.DataFrame(patch), width="stretch")
        render_plot(plot_patch_heatmap(patch, "Numerical patch heatmap"))
        st.subheader("Matrix experiment")
        operation = st.radio("Operation", ["A + c", "cA"], horizontal=True)
        scalar = st.slider("Scalar c", -255.0, 255.0, 1.0, 1.0)
        transformed = np.clip(
            channels[0] + scalar if operation == "A + c" else channels[0] * scalar,
            0,
            255,
        )
        render_plot(plot_image_matrix(transformed, "Scalar matrix operation"))
        with st.expander("📊 View Complete Image Matrix (Scrollable Table)"):
            st.dataframe(pd.DataFrame(channels[0]), height=500)

    elif stage.startswith("2."):
        started = time.perf_counter()
        svds = compute_channel_svds(channels)
        elapsed_ms = (time.perf_counter() - started) * 1000
        U, S, Vt = svds[0]
        st.write(f"SVD dimensions: U {U.shape}, S {S.shape}, Vᵀ {Vt.shape}")
        st.metric("Live SVD execution", f"{elapsed_ms:.3f} ms")
        render_plot(plot_singular_vectors(U, Vt, 5))
        components = [rank_one_component(U, S, Vt, i) for i in range(min(6, len(S)))]
        render_plot(plot_rank_one_components(components, 6))

    elif stage.startswith("3."):
        U, S, Vt = svds[0]
        k_marker = st.slider("Marker rank", 1, len(S), min(20, len(S)))
        render_plot(plot_scree(S, k_marker))
        render_plot(plot_cumulative_energy(S, k_marker, [0.90, 0.95, 0.99]))

    elif stage.startswith("4."):
        U, S, Vt = svds[0]
        target = st.select_slider(
            "Target energy", options=[0.90, 0.95, 0.99], value=0.95
        )
        target_rank = rank_for_energy(S, target)
        k = st.slider("Rank k", 1, len(S), min(target_rank, len(S)))
        reconstructed = channel_reconstruct(svds, k)
        render_plot(plot_reconstruction_comparison(original, reconstructed))
        ranks = [r for r in [1, 2, 5, 10, 20, 50, 100] if r <= len(S)]
        film_images = [channel_reconstruct(svds, r) for r in ranks]
        render_plot(plot_filmstrip(film_images, ranks))
        Uk, Sk, Vtk = channel_slice(svds[0], k)
        Ak = reconstruct(Uk, Sk, Vtk)
        theoretical_f = frobenius_error(S, k)
        measured_f = np.sqrt(np.sum((channels[0] - Ak) ** 2))
        theoretical_s = spectral_error(S, k)
        measured_s = np.linalg.norm(channels[0] - Ak, ord=2)
        st.write(
            pd.DataFrame(
                {
                    "Metric": ["Frobenius error", "Spectral error"],
                    "Theoretical": [theoretical_f, theoretical_s],
                    "Measured": [float(measured_f), float(measured_s)],
                }
            )
        )

    elif stage.startswith("5."):
        U, S, Vt = svds[0]
        m, n = channels[0].shape
        k = st.slider("Quantization rank k", 1, len(S), min(20, len(S)))
        theoretical = theoretical_storage(m, n, k) * len(channels) * 8
        raw = m * n * len(channels) * 8
        breakeven = break_even_rank(m, n)
        st.write(
            {
                "theoretical float64 factor bytes": theoretical,
                "original float64 image bytes": raw,
                "break-even rank": breakeven,
            }
        )
        if k >= breakeven:
            st.warning("This rank is at or above the storage break-even point.")
        quantized = quantize_factors(*channel_slice(svds[0], k), bits=8)
        payload = serialize_compressed(quantized, channels[0].shape)
        Uq, Sq, Vtq = dequantize_factors(quantized)
        reconstructed = np.clip(reconstruct(Uq, Sq, Vtq), 0, 255)
        ratio = compression_ratio(original.nbytes, len(payload))
        psnr_value = psnr(original, reconstructed)
        ssim_value = ssim(original, reconstructed)
        st.write(
            {
                "measured .npz bytes": len(payload),
                "real compression ratio": ratio,
                "PSNR": psnr_value,
                "SSIM": ssim_value,
            }
        )
        st.download_button("Download compressed .npz", payload, "svd-compressed.npz")
        pil = Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8), mode="L")
        png = io.BytesIO()
        pil.save(png, format="PNG")
        st.download_button(
            "Download reconstructed PNG", png.getvalue(), "reconstructed.png"
        )

    elif stage.startswith("6."):
        digest = image_digest(original)
        key = f"quality_{digest}"
        if key not in st.session_state:
            st.session_state[key] = quality_analysis(svds, channels)
        ranks, psnrs, ssims, svd_sizes, _, jpeg_sizes, jpeg_psnrs = st.session_state[
            key
        ]
        render_plot(plot_psnr_ssim_vs_rank(ranks, psnrs, ssims))
        render_plot(plot_rate_distortion(svd_sizes, psnrs, jpeg_sizes, jpeg_psnrs))
        st.info(
            "JPEG uses a fixed local DCT basis and, for natural photographs, "
            "often allocates storage more efficiently than a global, "
            "data-dependent SVD basis."
        )

    else:
        noise_sigma = st.slider("Additive Gaussian noise σ", 0.0, 80.0, 20.0, 1.0)
        noisy, denoised, residual, thresholds = compute_denoising(
            channels, noise_sigma, "gavish-donoho"
        )
        if len(noisy) == 1:
            noisy_image, denoised_image, residual_image = (
                noisy[0],
                denoised[0],
                residual[0],
            )
        else:
            noisy_image = np.clip(np.stack(noisy, axis=2), 0, 255)
            denoised_image = np.clip(np.stack(denoised, axis=2), 0, 255)
            residual_image = np.stack(residual, axis=2)
        render_plot(
            plot_denoise_comparison(noisy_image, denoised_image, residual_image)
        )
        U, S, Vt = svds[0]
        threshold, omega = gavish_donoho_threshold(
            S, channels[0].shape[0], channels[0].shape[1]
        )
        render_plot(plot_spectrum_threshold(S, threshold))
        st.write(
            {
                "Gavish-Donoho threshold": thresholds[0],
                "omega": omega,
                "noise sigma": noise_sigma,
            }
        )


if __name__ == "__main__":
    main()
