from __future__ import annotations

import io
from typing import Any

import numpy as np


def quantize_factors(
    U_k: np.ndarray, S_k: np.ndarray, Vt_k: np.ndarray, bits: int = 8
) -> dict[str, np.ndarray | int]:
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    maximum = (1 << bits) - 1
    su = np.max(np.abs(U_k), axis=0)
    su = np.where(su == 0, 1.0, su)
    sv = np.max(np.abs(Vt_k), axis=1)
    sv = np.where(sv == 0, 1.0, sv)
    u_quant = np.rint(U_k / su * maximum).astype(np.int16)
    vt_quant = np.rint(Vt_k / sv[:, None] * maximum).astype(np.int16)
    return {
        "U": u_quant,
        "S": np.asarray(S_k, dtype=np.float32),
        "Vt": vt_quant,
        "su": su.astype(np.float64),
        "sv": sv.astype(np.float64),
        "bits": bits,
    }


def dequantize_factors(
    quantized: dict[str, np.ndarray | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = (1 << int(quantized["bits"])) - 1
    su = np.asarray(quantized["su"], dtype=np.float64)
    sv = np.asarray(quantized["sv"], dtype=np.float64)
    U = np.asarray(quantized["U"], dtype=np.float64) * su / maximum
    S = np.asarray(quantized["S"], dtype=np.float64)
    Vt = np.asarray(quantized["Vt"], dtype=np.float64) * sv[:, None] / maximum
    return U, S, Vt


def serialize_compressed(
    quantized: dict[str, np.ndarray | int],
    original_shape: tuple[int, int],
    dtype: Any = np.float64,
) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        U=quantized["U"],
        S=quantized["S"],
        Vt=quantized["Vt"],
        su=quantized["su"],
        sv=quantized["sv"],
        bits=np.asarray(quantized["bits"], dtype=np.int16),
        original_shape=np.asarray(original_shape, dtype=np.int64),
        dtype=np.asarray(str(dtype)),
    )
    return buffer.getvalue()


def deserialize_compressed(
    data: bytes,
) -> tuple[dict[str, np.ndarray | int], tuple[int, int]]:
    buffer = io.BytesIO(data)
    with np.load(buffer, allow_pickle=False) as archive:
        quantized: dict[str, np.ndarray | int] = {
            "U": archive["U"],
            "S": archive["S"],
            "Vt": archive["Vt"],
            "su": archive["su"],
            "sv": archive["sv"],
            "bits": int(archive["bits"].item()),
        }
        shape = tuple(int(v) for v in archive["original_shape"])
    return quantized, (shape[0], shape[1])


def theoretical_storage(m: int, n: int, k: int) -> int:
    return k * (m + n + 1)


def break_even_rank(m: int, n: int) -> float:
    return (m * n) / float(m + n + 1)
