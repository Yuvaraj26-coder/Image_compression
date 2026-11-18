# SVD Image Compressor

An interactive Streamlit walkthrough of low-rank image compression using the
singular value decomposition.

## Run

```bash
python -m pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

Open `http://127.0.0.1:8501`.

## Mathematics

For an image matrix \(A \in \mathbb{R}^{m \times n}\), the SVD is

\[
A = U\Sigma V^T.
\]

The rank-\(k\) approximation is \(A_k=U_k\Sigma_kV_k^T\). By the Eckart-Young-
Mirsky theorem it minimizes both the Frobenius and spectral approximation errors
among all rank-\(k\) matrices:

\[
\|A-A_k\|_F=\sqrt{\sum_{i=k+1}^{r}\sigma_i^2},
\qquad
\|A-A_k\|_2=\sigma_{k+1}.
\]

Theoretical storage for float factors is \(k(m+n+1)\) scalar values. The eight-bit
mode quantizes each factor column or row using a scale, serializes a compressed
`.npz`, and reports measured file size and real reconstruction quality.

Gavish-Donoho denoising uses

\[
\tau=\omega(\beta)\operatorname{median}(S),\qquad
\beta=\frac{\min(m,n)}{\max(m,n)},
\]
\[
\omega(\beta)\approx0.56\beta^3-0.95\beta^2+1.82\beta+1.43.
\]

## Test

```bash
pytest tests/ -v
ruff check .
black --check .
mypy svdlab/
```
