import io

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from svdlab.io import load_image, resize_image, validate_image


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_loads_common_modes(mode):
    source = io.BytesIO()
    if mode == "L":
        img = Image.fromarray(np.ones((20, 30), dtype=np.uint8), mode="L")
    elif mode == "P":
        img = Image.fromarray(np.ones((20, 30), dtype=np.uint8), mode="P")
    elif mode == "RGBA":
        img = Image.fromarray(np.ones((20, 30, 4), dtype=np.uint8), mode="RGBA")
    else:
        img = Image.fromarray(np.ones((20, 30, 3), dtype=np.uint8), mode="RGB")
    img.save(source, format="PNG")
    source.seek(0)
    array, info = load_image(source, max_side=512)
    assert array.dtype == np.float64
    assert array.shape[0] > 0
    assert info["mode"] == mode


def test_validate_grayscale():
    validate_image(np.zeros((5, 5)))


def test_validate_rgb():
    validate_image(np.zeros((5, 5, 3)))


@pytest.mark.parametrize(
    "bad", [None, np.array([]), np.zeros((2, 2, 2, 2)), np.array([[np.nan]])]
)
def test_validate_rejects_bad_images(bad):
    with pytest.raises((TypeError, ValueError)):
        validate_image(bad)


def test_resize_large_array():
    array = np.ones((1000, 800, 3), dtype=np.float64)
    resized, was_resized = resize_image(array, 200)
    assert was_resized
    assert max(resized.shape[:2]) == 200


def test_resize_small_array_no_change():
    array = np.ones((20, 30), dtype=np.float64)
    resized, was_resized = resize_image(array, 512)
    assert not was_resized
    assert np.array_equal(resized, array)


def test_corrupted_file_raises():
    with pytest.raises(UnidentifiedImageError):
        load_image(io.BytesIO(b"not an image"), max_side=512)


def test_samples_load():
    for name in ["camera", "astronaut", "coffee"]:
        array, _ = load_image("sample", name, 128)
        assert array.size > 0
