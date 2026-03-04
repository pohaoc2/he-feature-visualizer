import numpy as np


def test_tissue_mask_detects_colored_region():
    """
    Contract:
    - `tissue_mask_rgb(rgb)` returns a boolean mask with same HxW
    - It should label a colored (non-white) blob as tissue, and white background as non-tissue
    """
    h, w = 64, 64
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[16:48, 16:48, :] = np.array([180, 60, 120], dtype=np.uint8)  # tissue-like

    from tissue_mask import tissue_mask_rgb  # noqa: WPS433

    m = tissue_mask_rgb(rgb)
    assert m.shape == (h, w)
    assert m.dtype == bool

    # Center should be tissue; corner should be background
    assert bool(m[32, 32]) is True
    assert bool(m[0, 0]) is False


def test_tissue_mask_rejects_gray_dirty_background():
    """
    Dirty non-white background (light gray) should still be background.
    """
    h, w = 64, 64
    rgb = np.full((h, w, 3), 235, dtype=np.uint8)  # slightly gray background
    rgb[20:44, 20:44, :] = np.array([170, 80, 120], dtype=np.uint8)

    from tissue_mask import tissue_mask_rgb  # noqa: WPS433

    m = tissue_mask_rgb(rgb)
    assert bool(m[0, 0]) is False
    assert bool(m[32, 32]) is True

