import time
import warnings

import cv2 as cv
import numpy as np
import pytest

from simple_image_inpainter import (
    ComplexBackgroundError,
    ComplexBackgroundWarning,
    SimpleImageInpainter,
)


def _rect_mask(shape, x0, y0, x1, y1):
    mask = np.zeros(shape[:2], np.uint8)
    cv.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def _gradient(h=300, w=400, lo=200, hi=240, noise=0.0, seed=0):
    row = np.linspace(lo, hi, w, dtype=np.float32)
    img = np.dstack([np.tile(row, (h, 1))] * 3)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- backgrounds

def test_flat_background_is_restored_exactly():
    img = np.full((200, 300, 3), (40, 120, 200), np.uint8)
    img[80:120, 100:160] = 0
    mask = _rect_mask(img.shape, 100, 80, 159, 119)
    result = SimpleImageInpainter(3, seed=0).inpaint(img, mask)
    assert (result == (40, 120, 200)).all()


def test_pixels_far_from_mask_are_untouched():
    img = _gradient(noise=2)
    mask = _rect_mask(img.shape, 150, 100, 200, 150)
    result = SimpleImageInpainter(3, seed=0).inpaint(img, mask)
    far = np.ones(mask.shape, bool)
    far[60:190, 110:240] = False
    assert (result[far] == img[far]).all()


def test_gradient_has_no_seam():
    img = _gradient(lo=100, hi=250)
    truth = img.copy()
    img[100:160, 150:250] = 0
    mask = _rect_mask(img.shape, 150, 100, 249, 159)
    result = SimpleImageInpainter(3, add_noise=False).inpaint(img, mask)
    err = np.abs(result.astype(int) - truth.astype(int))[100:160, 150:250]
    assert err.max() <= 2  # the old flat fill was off by ~30 here


def test_noise_is_reproduced_and_deterministic():
    img = _gradient(noise=4, seed=1)
    mask = _rect_mask(img.shape, 150, 100, 250, 200)
    a = SimpleImageInpainter(3, seed=7).inpaint(img, mask)
    b = SimpleImageInpainter(3, seed=7).inpaint(img, mask)
    assert (a == b).all()
    fill_std = a[110:190, 160:240].astype(float).std(axis=(0, 1)).mean()
    assert 2.5 < fill_std < 12  # plane trend + sigma~4 grain, not a flat plastic patch
    flat = SimpleImageInpainter(3, add_noise=False).inpaint(img, mask)
    assert flat[110:190, 160:240].astype(float).std(axis=(0, 1)).mean() < fill_std


def test_halo_around_tight_mask_is_covered_by_dilation():
    img = np.full((120, 120, 3), 230, np.uint8)
    cv.circle(img, (60, 60), 12, (20, 20, 20), -1)
    cv.circle(img, (60, 60), 14, (120, 120, 120), 2)  # halo ring outside the mask
    mask = np.zeros((120, 120), np.uint8)
    cv.circle(mask, (60, 60), 12, 255, -1)
    result = SimpleImageInpainter(4, add_noise=False).inpaint(img, mask)
    assert (result == 230).all()


# ------------------------------------------------------------------- multiple

def test_neighbouring_regions_do_not_contaminate_each_other():
    img = np.full((100, 200, 3), 255, np.uint8)
    img[40:60, 50:80] = 0
    img[40:60, 83:113] = 0  # 3 px gap from the first object
    mask = np.zeros((100, 200), np.uint8)
    mask[40:60, 50:80] = 255
    mask[40:60, 83:113] = 255
    result = SimpleImageInpainter(2, add_noise=False).inpaint(img, mask)
    assert (result == 255).all()


def test_two_color_boundary_uses_nearest_not_a_blend():
    img = np.full((200, 300, 3), 50, np.uint8)
    img[:, 150:] = 200
    truth = img.copy()
    img[80:120, 120:180] = (0, 0, 255)
    mask = _rect_mask(img.shape, 120, 80, 179, 119)
    inp = SimpleImageInpainter(3, seed=0)
    with pytest.warns(ComplexBackgroundWarning):
        result, reports = inp.inpaint_with_report(img, mask)
    assert reports[0].method == 'nearest' and reports[0].is_complex
    region = (slice(80, 120), slice(120, 180))
    close = (np.abs(result[region].astype(int) - truth[region].astype(int)).max(axis=2) <= 3).mean()
    assert close > 0.9
    with pytest.warns(ComplexBackgroundWarning):
        telea = SimpleImageInpainter(3, on_complex='telea').inpaint(img, mask)
    err_nearest = np.abs(result[region].astype(int) - truth[region].astype(int)).mean()
    err_telea = np.abs(telea[region].astype(int) - truth[region].astype(int)).mean()
    assert err_nearest < err_telea


def test_nearest_matches_bruteforce_on_random_ring():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    mask = _rect_mask(img.shape, 30, 30, 50, 50)
    inp = SimpleImageInpainter(0, feather=0, ring_width=3, on_complex='nearest')
    with pytest.warns(ComplexBackgroundWarning):
        result = inp.inpaint(img, mask)
    ring = np.zeros(mask.shape, bool)
    ring[27:54, 27:54] = True
    ring[30:51, 30:51] = False
    ry, rx = np.nonzero(ring)
    hits, total = 0, 0
    for y in range(30, 51):
        for x in range(30, 51):
            d = (ry - y) ** 2 + (rx - x) ** 2
            best = img[ry[d == d.min()], rx[d == d.min()]]
            hits += any((result[y, x] == b).all() for b in best)
            total += 1
    assert hits / total > 0.9  # DIST_MASK_5 is an approximation of L2


# ------------------------------------------------------------ complex handling

def _textured():
    rng = np.random.default_rng(5)
    return np.kron(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8), np.ones((4, 4, 1), np.uint8))


def test_complex_background_warns_and_reports():
    img = _textured()
    mask = _rect_mask(img.shape, 60, 40, 90, 70)
    with pytest.warns(ComplexBackgroundWarning):
        _, reports = SimpleImageInpainter().inpaint_with_report(img, mask)
    assert reports[0].is_complex and reports[0].complexity > 15


def test_on_complex_raise_and_skip():
    img = _textured()
    mask = _rect_mask(img.shape, 60, 40, 90, 70)
    with pytest.raises(ComplexBackgroundError):
        SimpleImageInpainter(on_complex='raise').inpaint(img, mask)
    with pytest.warns(ComplexBackgroundWarning):
        result = SimpleImageInpainter(on_complex='skip').inpaint(img, mask)
    assert (result == img).all()


@pytest.mark.parametrize('mode', ['telea', 'ns'])
def test_cv_fallbacks_run_for_uint16_and_bgra(mode):
    img = np.dstack([_textured().astype(np.uint16) * 257, np.full((120, 160), 65535, np.uint16)])
    mask = _rect_mask(img.shape, 60, 40, 90, 70)
    with pytest.warns(ComplexBackgroundWarning):
        result = SimpleImageInpainter(on_complex=mode).inpaint(img, mask)
    assert result.shape == img.shape and result.dtype == np.uint16


def test_no_warning_on_simple_background():
    img = _gradient(noise=2)
    mask = _rect_mask(img.shape, 100, 100, 140, 140)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        SimpleImageInpainter().inpaint(img, mask)


# ------------------------------------------------------------- formats / edges

def test_bgra_keeps_alpha():
    img = np.dstack([_gradient(noise=2), np.full((300, 400), 255, np.uint8)])
    mask = _rect_mask(img.shape, 100, 100, 140, 140)
    result = SimpleImageInpainter(seed=0).inpaint(img, mask)
    assert result.shape == img.shape and (result[..., 3] == 255).all()


def test_grayscale_2d_and_uint16():
    for dtype, value in ((np.uint8, 120), (np.uint16, 30000)):
        img = np.full((100, 100), value, dtype)
        img[40:60, 40:60] = 0
        result = SimpleImageInpainter(2).inpaint(img, _rect_mask(img.shape, 40, 40, 59, 59))
        assert result.shape == (100, 100) and result.dtype == dtype and (result == value).all()


def test_region_touching_image_border():
    img = _gradient()
    truth = img.copy()
    img[:30, :40] = 0
    result = SimpleImageInpainter(3, add_noise=False).inpaint(img, _rect_mask(img.shape, 0, 0, 39, 29))
    assert np.abs(result.astype(int) - truth.astype(int))[:30, :40].max() <= 3


def test_bool_and_soft_masks():
    img = np.full((100, 100, 3), 90, np.uint8)
    img[40:60, 40:60] = 0
    hard = _rect_mask(img.shape, 40, 40, 59, 59)
    soft = cv.GaussianBlur(hard, (5, 5), 0)  # antialiased edge, not strictly binary
    for m in (hard > 0, soft):
        assert (SimpleImageInpainter(2).inpaint(img, m) == 90).all()


def test_input_not_modified_and_empty_mask_is_copy():
    img = _gradient(noise=2)
    before = img.copy()
    mask = _rect_mask(img.shape, 100, 100, 140, 140)
    SimpleImageInpainter().inpaint(img, mask)
    assert (img == before).all()
    empty = SimpleImageInpainter().inpaint(img, np.zeros(mask.shape, np.uint8))
    assert (empty == img).all() and empty is not img


def test_input_validation():
    img = np.zeros((50, 50, 3), np.uint8)
    inp = SimpleImageInpainter()
    with pytest.raises(ValueError):
        inp.inpaint(img, np.zeros((40, 50), np.uint8))
    with pytest.raises(ValueError):
        inp.inpaint(img, np.zeros((50, 50, 3), np.uint8))
    with pytest.raises(TypeError):
        inp.inpaint(img.astype(np.float32), np.zeros((50, 50), np.uint8))
    with pytest.raises(TypeError):
        inp.inpaint([[0]], np.zeros((50, 50), np.uint8))
    with pytest.raises(ValueError):
        SimpleImageInpainter(on_complex='bogus')
    with pytest.raises(ValueError):
        SimpleImageInpainter(dilation_size=-1)


def test_reports_describe_regions():
    img = np.full((100, 200, 3), 128, np.uint8)
    mask = np.zeros((100, 200), np.uint8)
    mask[10:20, 10:20] = 255
    mask[60:80, 150:180] = 255
    _, reports = SimpleImageInpainter().inpaint_with_report(img, mask)
    assert [r.bbox for r in reports] == [(10, 10, 10, 10), (150, 60, 30, 20)]
    assert all(r.method == 'flat' and not r.is_complex for r in reports)


def test_many_small_regions_are_fast():
    rng = np.random.default_rng(0)
    img = np.full((2000, 2000, 3), 200, np.uint8)
    mask = np.zeros((2000, 2000), np.uint8)
    for x, y in rng.integers(20, 1980, (300, 2)):
        cv.circle(mask, (int(x), int(y)), 4, 255, -1)
    start = time.perf_counter()
    SimpleImageInpainter().inpaint(img, mask)
    assert time.perf_counter() - start < 3.0  # the previous implementation needed ~7 s
