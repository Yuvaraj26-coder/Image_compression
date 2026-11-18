# Master Agent Specification: Reconstruct the SVD Image Compressor

You are a principal Python software engineer, scientific visualization specialist, and applied mathematician. Your mission is to reconstruct from scratch the complete, production-grade, portfolio-ready repository:

> **"SVD Image Compressor: An Interactive Walkthrough of Low-Rank Image Compression"**

Every number presented in the application must be computed live from real mathematics; placeholder text, mock data, and fabricated metrics are strictly forbidden. The system must run flawlessly on modern Python (≥ 3.10) across Linux, macOS, and Windows without network latency, WebSocket timeouts, or memory leaks.

---

## 1. Project Directory Structure

Reconstruct the repository using the exact layout below:

```text
svd-image-compressor/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated test matrix (Python 3.10 & 3.12)
├── .streamlit/
│   └── config.toml              # Server binding, WebSocket & loopback stability
├── notebooks/
│   └── theory_walkthrough.ipynb # Executable Jupyter notebook with real outputs
├── svdlab/                      # Core computational library (Zero Streamlit imports)
│   ├── __init__.py              # Public library exports
│   ├── io.py                    # Image loading, preprocessing, validation, resizing
│   ├── decompose.py             # SVD, truncation, rank-1 components, energy
│   ├── compress.py              # 8-bit quantization, serialization, storage math
│   ├── metrics.py               # MSE, PSNR, SSIM, Frobenius and spectral errors
│   └── plots.py                 # Matplotlib figure generators (Agg backend)
├── tests/                       # Comprehensive pytest suite
│   ├── __init__.py
│   ├── test_app.py              # AppTest stage smoke tests & UI logic
│   ├── test_compress.py         # Serialization, quantization, storage tests
│   ├── test_decompose.py        # SVD, truncation, Eckart-Young theorem tests
│   ├── test_denoise.py          # Gavish-Donoho threshold rank recovery tests
│   ├── test_io.py               # Image format, mode conversions & resize tests
│   └── test_metrics.py          # Numerical accuracy & underflow prevention tests
├── app.py                       # High-performance Streamlit UI entry point
├── pyproject.toml               # Ruff, black, mypy & pytest configuration
├── setup.py                     # Setuptools packaging for editable installs
├── requirements.txt             # Pinned dependency boundaries
├── LICENSE                      # MIT License
└── README.md                    # Project documentation with mathematical rigor
```

---

## 2. Configuration & Packaging Files

### `requirements.txt`
```text
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
Pillow>=10.0.0
matplotlib>=3.7.0
pandas>=2.0.0
pytest>=7.4.0
ruff>=0.1.0
black>=23.0.0
mypy>=1.5.0
```

### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="svdlab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.10",
)
```

### `pyproject.toml`
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4"]

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### `.streamlit/config.toml`
> **CRITICAL REQUIREMENT**: To prevent Windows IPv6 loopback collisions (`::1` vs `127.0.0.1`), telemetry connection delays, and browser WebSocket dropouts ("Connection timed out"), configure:
```toml
[server]
address = "127.0.0.1"
port = 8501
headless = false
enableCORS = false
enableXsrfProtection = false
maxMessageSize = 500

[browser]
serverAddress = "127.0.0.1"
gatherUsageStats = false
```

### `.github/workflows/ci.yml`
```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Lint with ruff
      run: ruff check .
    
    - name: Format check with black
      run: black --check .
    
    - name: Type check with mypy
      run: mypy svdlab/
    
    - name: Run tests
      run: pytest tests/ -v
```

---

## 3. Computational Engine Specification (`svdlab/`)

> **Architectural Constraint**: Under no circumstances should any module in `svdlab/` import `streamlit`. Keep core computations strictly decoupled and independently testable.

### 3.1 `svdlab/__init__.py`
Export all primary functions:
```python
from svdlab.io import load_image, validate_image, resize_image
from svdlab.decompose import (
    compute_svd,
    truncate_svd,
    rank_one_component,
    reconstruct,
    cumulative_energy,
    rank_for_energy,
)
from svdlab.compress import (
    quantize_factors,
    dequantize_factors,
    serialize_compressed,
    deserialize_compressed,
    theoretical_storage,
    break_even_rank,
)
from svdlab.metrics import (
    mse,
    psnr,
    ssim,
    compression_ratio,
    frobenius_error,
    spectral_error,
)
from svdlab.plots import (
    plot_image_matrix,
    plot_patch_heatmap,
    plot_singular_vectors,
    plot_rank_one_components,
    plot_scree,
    plot_cumulative_energy,
    plot_reconstruction_comparison,
    plot_filmstrip,
    plot_error_heatmap,
    plot_psnr_ssim_vs_rank,
    plot_rate_distortion,
    plot_denoise_comparison,
    plot_spectrum_threshold,
)
```

### 3.2 `svdlab/io.py`
Implement image ingestion and normalization:
- **`load_image(source="sample", sample_name="camera", max_side=512)`**:
  - Sample images: `"camera"`, `"astronaut"`, `"coffee"` via `skimage.data`.
  - Uploaded files: Open via `PIL.Image.open(source)`.
  - Downscale while still in Pillow using `pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)` before allocating NumPy arrays.
  - Flatten transparent modes (`"RGBA"`, `"LA"`, `"P"`) onto an opaque white background `(255, 255, 255)`.
  - Convert `"CMYK"` and non-RGB/L modes cleanly to `"RGB"`.
  - Return `(image_array, info_dict)` where `image_array` is `float64` in range $[0, 255]$.
- **`validate_image(img_array)`**:
  - Verify array is `np.ndarray`, dimension is 2 or 3, size $> 0$, and all entries are finite (`np.isfinite`).
- **`resize_image(img_array, max_side=512)`**:
  - Rescales array if $\max(h, w) > \text{max\_side}$ using Pillow `LANCZOS` filter and returns `(resized_array, was_resized)`.

### 3.3 `svdlab/decompose.py`
Implement SVD and rank operations:
- **`compute_svd(matrix, full_matrices=False)`**:
  - Enforce 2D array. Use `np.linalg.svd(matrix, full_matrices=False)`.
  - Return `(U, S, Vt)` where $U \in \mathbb{R}^{m \times r}$, $S \in \mathbb{R}^{r}$, $V^T \in \mathbb{R}^{r \times n}$, with $r = \min(m, n)$.
- **`truncate_svd(U, S, Vt, k)`**:
  - Clamp $k \in [1, \text{len}(S)]$.
  - Return $U[:, :k]$, $S[:k]$, $V^T[:k, :]$.
- **`rank_one_component(U, S, Vt, i)`**:
  - Compute outer product: $A_i = \sigma_i (u_i \otimes v_i^T) = \sigma_i u_i v_i^T$.
- **`reconstruct(U_k, S_k, Vt_k)`**:
  - Compute matrix product: $U_k \operatorname{diag}(S_k) V_k^T$.
- **`cumulative_energy(S)`**:
  - Energy $E_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^r \sigma_j^2}$.
  - **Zero Division Guard**: If $\sum \sigma_j^2 \le 0$, return `np.zeros_like(S)`.
- **`rank_for_energy(S, target_energy=0.95)`**:
  - Return the minimal integer $k \ge 1$ such that $E_k \ge \text{target\_energy}$ using `np.searchsorted`.

### 3.4 `svdlab/compress.py`
Implement quantization and true file serialization:
- **`quantize_factors(U_k, S_k, Vt_k, bits=8)`**:
  - Column-wise scaling for $U_k$: $s_u = \max(|U_k|, \text{axis}=0)$ (guard zeros with 1).
  - Row-wise scaling for $V_k^T$: $s_v = \max(|V_k^T|, \text{axis}=1)$ (guard zeros with 1).
  - Quantize: $\text{round}\left(\frac{U_k}{s_u} (2^{\text{bits}} - 1)\right)$ as `np.int16`.
  - Keep singular values $S_k$ in `float32`.
- **`dequantize_factors(quantized)`**:
  - Reconstruct factors in `float64`: $\frac{\text{quantized} \times \text{scales}}{2^{\text{bits}} - 1}$.
- **`serialize_compressed(quantized, original_shape, dtype=np.float64)`**:
  - Serialize using `np.savez_compressed` into `io.BytesIO()`.
- **`deserialize_compressed(data)`**:
  - Read from `io.BytesIO(data)` via `np.load` and restore structures.
- **`theoretical_storage(m, n, k)`**:
  - Formula: $k(m + n + 1)$.
- **`break_even_rank(m, n)`**:
  - Formula: $k_{\text{break-even}} = \frac{m \cdot n}{m + n + 1}$.

### 3.5 `svdlab/metrics.py`
Implement image error and quality metrics:
- **`mse(image1, image2)`**:
  - **Underflow Protection**: Cast inputs to `np.float64` before subtraction: `(img1.astype(np.float64) - img2.astype(np.float64))**2`.
- **`psnr(image1, image2, data_range=255.0)`**:
  - Formula: $20 \log_{10}(\text{data\_range}) - 10 \log_{10}(\text{MSE})$. If $\text{MSE} == 0$, return `float('inf')`.
- **`ssim(image1, image2, data_range=255.0)`**:
  - Wrap `skimage.metrics.structural_similarity` with `channel_axis=2` for 3D arrays.
- **`compression_ratio(original_size, compressed_size)`**:
  - Return $\frac{\text{original\_size}}{\text{compressed\_size}}$.
- **`frobenius_error(S, k)`**:
  - Theoretical Eckart-Young-Mirsky Frobenius error: $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$.
- **`spectral_error(S, k)`**:
  - Theoretical Eckart-Young-Mirsky spectral error: $\|A - A_k\|_2 = \sigma_{k+1}$ (return $0.0$ if $k \ge \text{len}(S)$).

### 3.6 `svdlab/plots.py`
All visualization builders using `matplotlib.use('Agg')`:
- `plot_image_matrix(image, title)`
- `plot_patch_heatmap(patch, title)` (with labeled numeric cell annotations)
- `plot_singular_vectors(U, Vt, n_vectors=5)` (dual subplots for left and right vectors)
- `plot_rank_one_components(components, n_components=6)` (`RdBu_r` diverging colormap)
- `plot_scree(S, k=None)` (log scale singular values)
- `plot_cumulative_energy(S, k=None, targets=None)` (percentage annotations)
- `plot_reconstruction_comparison(original, reconstructed)` (original, reconstruction, absolute error heatmap)
- `plot_filmstrip(images, k_values)` (horizontal multi-panel progression)
- `plot_error_heatmap(original, reconstructed)`
- `plot_psnr_ssim_vs_rank(ranks, psnrs, ssims, current_k=None)` (twin $y$-axes)
- `plot_rate_distortion(svd_sizes, svd_psnrs, jpeg_sizes, jpeg_psnrs)` (bits/file size vs PSNR curve)
- `plot_denoise_comparison(noisy, denoised, residual)` (**`vmin == vmax` guard**: if max absolute residual is 0, default to 1.0)
- `plot_spectrum_threshold(S, threshold)`

---

## 4. Streamlit Application Architecture (`app.py`)

### 4.1 Lifecycle & Caching Rules
1. **`st.set_page_config()`** MUST be called as the very first line inside `main()`. Do not call it at module level.
2. **Preventing WebSocket Disconnects**:
   - Cache sample and upload loading via `@st.cache_data(show_spinner=False)`.
   - Cache channel SVD decomposition:
     ```python
     @st.cache_data(show_spinner=False)
     def compute_channel_svds(channel_images):
         return [compute_svd(channel) for channel in channel_images]
     ```
   - Cache Gavish-Donoho denoising:
     ```python
     @st.cache_data(show_spinner=False)
     def compute_denoising(channel_images, noise_sigma, mode):
         ...
     ```
   - Cache Quality Analysis (rate-distortion & PSNR curves) on a log-spaced grid of 40 ranks inside `st.session_state` keyed by an image byte digest.
3. **Memory Management**:
   - Wrap every Matplotlib display with:
     ```python
     def render_plot(fig):
         st.pyplot(fig)
         plt.close(fig)
     ```
4. **Color Processing Modes**:
   - **Grayscale**: Matrix shape $(m, n)$.
   - **Color (YCbCr)**: Decoupled luminance ($Y$) and chrominance ($Cb, Cr$) channels.
   - **Color (RGB per channel)**: Independent decomposition on $R, G, B$ planes.

### 4.2 Interactive Walkthrough Stages
The main UI must feature a horizontal radio stepper containing the 6 guided stages:

1. **Stage 1: Image as a Matrix**
   - Display image and live metadata: shape, dtype, element count, raw bytes.
   - Zoom into an 8×8 patch (using `min(8, h), min(8, w)` bounds).
   - Display the patch as an interactive numerical table and an annotated heatmap.
   - **Complete Matrix Viewer**: Provide an expandable container (`st.expander("📊 View Complete Image Matrix (Scrollable Table)")`) allowing users to scroll through the full numerical matrix of any channel.
   - Interactive Matrix Experiment: Allow scalar operations ($A + c$ and $cA$) with clipping $[0, 255]$ to demonstrate matrix arithmetic.

2. **Stage 2: The Decomposition**
   - Present $A = U \Sigma V^T$ with live dimensions and SVD execution duration in milliseconds.
   - Plot first 5 left and right singular vectors.
   - Render the first 6 rank-1 components ($\sigma_i u_i v_i^T$) as individual heatmap matrices.

3. **Stage 3: Where the Information Lives**
   - Scree plot of singular values on a logarithmic scale with marker at current $k$.
   - Cumulative energy curve $\sum \sigma_i^2 / \sum \sigma_j^2$ with markers indicating ranks needed for 90%, 95%, and 99% energy.

4. **Stage 4: Building the Image Up**
   - Rank $k$ slider and target energy selector.
   - Three-panel comparison: Original | Reconstruction ($A_k$) | Absolute Error Heatmap.
   - Filmstrip showing reconstruction fidelity at ranks $k = [1, 2, 5, 10, 20, 50, 100]$.
   - Side-by-side verification of Eckart-Young-Mirsky theorem:
     - Theoretical Frobenius error $\sqrt{\sum_{i > k} \sigma_i^2}$ vs Measured $\|A - A_k\|_F$.
     - Theoretical Spectral error $\sigma_{k+1}$ vs Measured $\|A - A_k\|_2$.

5. **Stage 5: The Compression Maths**
   - Show theoretical storage: $k(m + n + 1)$ vs $m \cdot n$.
   - Show break-even rank $k < \frac{mn}{m + n + 1}$ and display a warning banner if $k \ge k_{\text{break-even}}$.
   - **Honest Float64 Reality Check**: Show that uncompressed float64 SVD factors consume 8× more bytes than 8-bit image pixels.
   - **8-Bit Quantized Storage**: Apply per-factor scale quantization, serialize via `.npz` (or multi-channel `.zip`), and display measured disk file size, real compression ratio, PSNR, and SSIM.
   - Download buttons for both the compressed `.npz`/`.zip` binary and the reconstructed PNG.

6. **Stage 6: Quality Analysis**
   - Dual-axis curve of PSNR and SSIM vs rank $k$.
   - Rate-distortion curve comparing SVD against JPEG (Pillow) across matched file sizes.
   - Plain-English mathematical explanation of why JPEG's local DCT basis outperforms SVD's global data-dependent basis on natural photographs.

7. **Bonus: Image Denoising**
   - Additive Gaussian noise slider ($\sigma$).
   - Gavish-Donoho optimal hard thresholding:
     $$\beta = \frac{\min(m, n)}{\max(m, n)}, \quad \omega(\beta) \approx 0.56\beta^3 - 0.95\beta^2 + 1.82\beta + 1.43, \quad \tau = \omega(\beta) \cdot \operatorname{median}(S)$$
   - Displays Noisy | Denoised | Residual alongside the thresholded singular value spectrum.

---

## 5. Automated Verification Test Suite (`tests/`)

The reconstructed test suite must contain at least 44 unit and integration tests passing with zero warnings:

- **`test_decompose.py`**:
  - Full-rank SVD reconstruction equals input within absolute tolerance $< 10^{-8}$.
  - Truncation shapes match rank $k$.
  - Rank-1 components sum exactly to rank-$k$ reconstruction.
  - Energy computation handles blank/zero matrices without division-by-zero warnings.
- **`test_metrics.py`**:
  - `uint8` subtraction underflow is prevented in MSE and PSNR.
  - Custom PSNR matches `skimage.metrics.peak_signal_noise_ratio` within $10^{-6}$.
  - Theoretical Frobenius and spectral errors match measured Euclidean differences within $10^{-8}$.
- **`test_compress.py`**:
  - Quantization $\to$ serialization $\to$ deserialization $\to$ dequantization round-trip maintains PSNR above 35 dB.
  - Storage formulas and break-even rank are validated for tall, wide, and square matrices.
- **`test_io.py`**:
  - Validates grayscale, RGB, RGBA, palette PNG, and corrupted files.
  - Verifies that large uploads are downscaled in Pillow before NumPy array allocation.
- **`test_denoise.py`**:
  - Gavish-Donoho threshold accurately recovers true underlying rank on a synthetic low-rank matrix with additive Gaussian noise.
- **`test_app.py`**:
  - Uses `streamlit.testing.v1.AppTest` to execute all 6 stages, matrix operations, and bonus denoising without throwing exceptions.

---

## 6. Execution & Verification Checklist

When reconstructing the repository, follow this exact sequence:

1. Create directory structure and configuration files (`.streamlit/config.toml`, `pyproject.toml`, `setup.py`, `requirements.txt`).
2. Implement `svdlab/` modules (`io.py`, `decompose.py`, `compress.py`, `metrics.py`, `plots.py`).
3. Install package in editable mode: `pip install -e .`.
4. Implement test suite under `tests/` and run `pytest`. Ensure all 44 tests pass.
5. Implement `app.py` with cached image I/O, responsive max side length slider (capped at 512 px), and full matrix explorer.
6. Verify startup stability: run `streamlit run app.py` and confirm clean binding to `http://127.0.0.1:8501`.
