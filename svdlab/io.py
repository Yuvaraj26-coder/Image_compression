from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from skimage import data


def validate_image(img_array: np.ndarray) -> None:
    if not isinstance(img_array, np.ndarray):
        raise TypeError("Image must be a NumPy array")
    if img_array.ndim not in (2, 3):
        raise ValueError("Image must be a 2D or 3D array")
    if img_array.size == 0:
        raise ValueError("Image array is empty")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("Image contains non-finite values")


def resize_image(img_array: np.ndarray, max_side: int = 512) -> tuple[np.ndarray, bool]:
    validate_image(img_array)
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    h, w = img_array.shape[:2]
    if max(h, w) <= max_side:
        return img_array.copy(), False
    scale = max_side / float(max(h, w))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64), True


def _flatten_pil(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode in ("RGBA", "LA", "P"):
        if pil_img.mode == "P":
            pil_img = pil_img.convert("RGBA")
        if pil_img.mode == "LA":
            pil_img = pil_img.convert("RGBA")
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(background, pil_img).convert("RGB")
    elif pil_img.mode == "CMYK":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    return pil_img


def load_image(
    source: Any = "sample", sample_name: str = "camera", max_side: int = 512
) -> tuple[np.ndarray, dict[str, Any]]:
    if source == "sample":
        samples = {
            "camera": data.camera,
            "astronaut": data.astronaut,
            "coffee": data.coffee,
        }
        if sample_name not in samples:
            raise ValueError(f"Unknown sample: {sample_name}")
        pil_img = Image.fromarray(samples[sample_name]())
    else:
        pil_img = Image.open(source)

    original_format = pil_img.format
    original_mode = pil_img.mode
    original_size = pil_img.size
    pil_img = _flatten_pil(pil_img)
    pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    array = np.asarray(pil_img, dtype=np.float64)
    validate_image(array)
    info: dict[str, Any] = {
        "format": original_format,
        "mode": original_mode,
        "original_size": original_size,
        "shape": array.shape,
        "downscaled": original_size != pil_img.size,
    }
    return array, info
