"""Inpainting of regions on simple backgrounds (flat, gradient, light noise).

For every connected region of the mask the background is estimated from a thin
ring of pixels *outside* the repaired area (a robust flat or planar colour
model), the region is refilled from that model with matching grain, and the
edge is feathered. If the ring turns out not to be a simple background, the
region is handled according to ``on_complex`` instead of being filled blindly.
"""
from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import cv2 as cv
import numpy as np

__all__ = [
    'SimpleImageInpainter',
    'ComponentReport',
    'ComplexBackgroundWarning',
    'ComplexBackgroundError',
]

_NDArray = np.ndarray
Model = Literal['auto', 'flat', 'plane']
OnComplex = Literal['nearest', 'telea', 'ns', 'skip', 'raise']

_MODELS = ('auto', 'flat', 'plane')
_ON_COMPLEX = ('nearest', 'telea', 'ns', 'skip', 'raise')
_INPAINT_FLAGS = {'telea': cv.INPAINT_TELEA, 'ns': cv.INPAINT_NS}

_MAD_TO_SIGMA = 1.4826      # MAD -> standard deviation for Gaussian data
_MIN_PLANE_SAMPLES = 10     # fewer ring pixels than this -> flat model only
_MAX_FIT_SAMPLES = 20_000   # ring pixels used for fitting (subsampled above)
_PLANE_GAIN = 0.8           # plane must cut residual to <80% of flat to win
_ALPHA_EPS = 1e-3


class ComplexBackgroundWarning(UserWarning):
    """Some regions have a background that is not simple (flat/gradient)."""


class ComplexBackgroundError(ValueError):
    """Raised with ``on_complex='raise'`` when a background is not simple."""


@dataclass(frozen=True)
class ComponentReport:
    """What happened to one connected region of the mask.

    method: 'flat' | 'plane' | 'nearest' | 'telea' | 'ns' | 'skipped'
    complexity: robust-model residual std over the ring, in 0..255 units
        (about 1-5 for clean/noisy simple backgrounds, tens for edges/texture).
    """
    label: int
    bbox: tuple[int, int, int, int]  # x, y, w, h of the original mask region
    method: str
    complexity: float
    is_complex: bool
    n_samples: int


@functools.lru_cache(maxsize=64)
def _ellipse(radius: int) -> _NDArray:
    size = 2 * radius + 1
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))


def _robust_fit(design: _NDArray, samples: _NDArray, unit: float) -> tuple[_NDArray, _NDArray]:
    """Least-squares fit with iterative outlier rejection. Returns (coef, inliers)."""
    n = len(samples)
    inliers = np.ones(n, dtype=bool)
    if design.shape[1] == 1:
        coef = np.median(samples, axis=0, keepdims=True)
    else:
        coef = np.linalg.lstsq(design, samples, rcond=None)[0]
    for _ in range(3):
        resid = samples - design @ coef
        center = np.median(resid[inliers], axis=0)
        sigma = _MAD_TO_SIGMA * np.median(np.abs(resid[inliers] - center), axis=0)
        limit = np.maximum(3.0 * sigma, 1.5 * unit)
        new = np.all(np.abs(resid - center) <= limit, axis=1)
        if new.sum() < min(_MIN_PLANE_SAMPLES, n) or np.array_equal(new, inliers):
            break
        inliers = new
        coef = np.linalg.lstsq(design[inliers], samples[inliers], rcond=None)[0]
    return coef, inliers


def _residual_std(design: _NDArray, samples: _NDArray, coef: _NDArray) -> float:
    return float(np.mean(np.std(samples - design @ coef, axis=0)))


class SimpleImageInpainter:
    """Inpainting for regions that sit on simple backgrounds.

    Args:
        dilation_size: Radius in pixels by which every mask region is grown.
            This covers anti-aliasing halos and shadows around the object.
        ring_width: Width in pixels of the background sampling ring, which
            lies just outside the repaired area.
        feather: Gaussian sigma (px) of the soft edge between the new fill
            and the original image. 0 disables feathering. The repaired
            area grows outward by ``ceil(2 * feather)`` px to make room.
        model: 'flat' (one colour), 'plane' (linear gradient) or 'auto'
            (plane only when it clearly beats flat).
        max_complexity: Ring residual std (0..255 scale) above which the
            background is considered not simple.
        on_complex: What to do with such regions: 'nearest' (colour of the
            nearest ring pixel, good for piecewise-flat backgrounds),
            'telea' / 'ns' (cv.inpaint), 'skip' (leave unchanged) or
            'raise' (ComplexBackgroundError). Unless 'raise', one
            ComplexBackgroundWarning is emitted per call.
        add_noise: Add Gaussian grain matching the ring noise so the fill
            does not look plastic.
        mask_threshold: Mask pixels with a value greater than this count as
            masked (bool masks are used as they are).
        seed: Seed for the grain, for reproducible output.
    """

    def __init__(
        self,
        dilation_size: int = 3,
        *,
        ring_width: int = 4,
        feather: float = 1.0,
        model: Model = 'auto',
        max_complexity: float = 15.0,
        on_complex: OnComplex = 'nearest',
        add_noise: bool = True,
        mask_threshold: int = 127,
        seed: int | None = None,
    ) -> None:
        if dilation_size < 0:
            raise ValueError('dilation_size must be >= 0')
        if ring_width < 1:
            raise ValueError('ring_width must be >= 1')
        if feather < 0:
            raise ValueError('feather must be >= 0')
        if model not in _MODELS:
            raise ValueError(f'model must be one of {_MODELS}')
        if on_complex not in _ON_COMPLEX:
            raise ValueError(f'on_complex must be one of {_ON_COMPLEX}')
        if max_complexity <= 0:
            raise ValueError('max_complexity must be > 0')
        self.radius = int(dilation_size)
        self.ring_width = int(ring_width)
        self.feather = float(feather)
        self.model = model
        self.max_complexity = float(max_complexity)
        self.on_complex = on_complex
        self.add_noise = add_noise
        self.mask_threshold = mask_threshold
        self._feather_px = math.ceil(2 * self.feather) if self.feather > 0 else 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API

    def inpaint(self, image: _NDArray, mask: _NDArray) -> _NDArray:
        """Return a copy of ``image`` with the masked regions repaired.

        ``image``: uint8/uint16, (H, W) or (H, W, 1/3/4).
        ``mask``: (H, W), bool or 0..255; values > ``mask_threshold`` are masked.
        """
        result, reports = self._run(image, mask)
        self._warn(reports)
        return result

    def inpaint_with_report(self, image: _NDArray, mask: _NDArray) -> tuple[_NDArray, list[ComponentReport]]:
        """Like :meth:`inpaint`, also returning one report per mask region."""
        result, reports = self._run(image, mask)
        self._warn(reports)
        return result, reports

    # ------------------------------------------------------------ internals

    @staticmethod
    def _prepare_inputs(image: _NDArray, mask: _NDArray, threshold: int) -> tuple[_NDArray, _NDArray]:
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError('image and mask must be numpy arrays')
        if image.dtype.type not in (np.uint8, np.uint16):
            raise TypeError(f'image dtype must be uint8 or uint16, got {image.dtype}')
        if image.ndim == 2:
            img = image[:, :, None]
        elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
            img = image
        else:
            raise ValueError(f'unsupported image shape {image.shape}; expected (H, W) or (H, W, 1/3/4)')
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError(f'mask must be single-channel (H, W), got shape {mask.shape}')
        if mask.shape != img.shape[:2]:
            raise ValueError(f'mask shape {mask.shape} does not match image shape {img.shape[:2]}')
        binary = mask if mask.dtype == bool else mask > threshold
        return img, np.ascontiguousarray(binary, dtype=np.uint8)

    def _run(self, image: _NDArray, mask: _NDArray) -> tuple[_NDArray, list[ComponentReport]]:
        img, binary = self._prepare_inputs(image, mask, self.mask_threshold)
        out = img.copy()
        reports: list[ComponentReport] = []
        if binary.any():
            height, width, _ = img.shape
            n, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
            reach = self.radius + self._feather_px
            pad = 2 * reach + self.ring_width + 2 + math.ceil(4 * self.feather)
            for label in range(1, n):
                x, y, w, h = (int(v) for v in stats[label, :4])
                x0, y0 = max(x - pad, 0), max(y - pad, 0)
                x1, y1 = min(x + w + pad, width), min(y + h + pad, height)
                reports.append(self._repair(
                    img[y0:y1, x0:x1], out[y0:y1, x0:x1], binary[y0:y1, x0:x1],
                    labels[y0:y1, x0:x1] == label, label, (x, y, w, h),
                ))
        return (out[:, :, 0] if image.ndim == 2 else out), reports

    def _repair(self, src: _NDArray, dst: _NDArray, mask_all: _NDArray, comp_bool: _NDArray,
                label: int, bbox: tuple[int, int, int, int]) -> ComponentReport:
        """Repair one region inside a cropped window. ``dst`` is a view into the output."""
        comp = comp_bool.astype(np.uint8)
        channels = src.shape[2]
        max_val = float(np.iinfo(src.dtype).max)
        unit = max_val / 255.0
        reach = self.radius + self._feather_px

        # Area to repaint (with room for the feather) and the sampling ring
        # outside it. Pixels of *any* mask region (plus its halo) are excluded.
        core = cv.dilate(comp, _ellipse(reach))
        forbidden = cv.dilate(mask_all, _ellipse(reach))
        ring_mask = (cv.dilate(comp, _ellipse(reach + self.ring_width)) > 0) & (forbidden == 0)
        ys, xs = np.nonzero(ring_mask)
        n_samples = len(ys)

        # Blend weight: hard inside `radius`, Gaussian falloff outside.
        alpha = core.astype(np.float32)
        if self.feather > 0:
            alpha = cv.GaussianBlur(alpha, (0, 0), self.feather)
        alpha = np.maximum(alpha, cv.dilate(comp, _ellipse(self.radius)).astype(np.float32))
        ty, tx = np.nonzero(alpha > _ALPHA_EPS)
        weight = alpha[ty, tx][:, None]

        method, complexity, pred = 'telea', float('inf'), None
        if n_samples:
            samples = src[ys, xs].astype(np.float64)
            roi_h, roi_w = src.shape[:2]
            scale = max(roi_h, roi_w) / 2.0

            def design(px: _NDArray, py: _NDArray) -> _NDArray:
                return np.column_stack([np.ones(len(px)), (px - roi_w / 2.0) / scale, (py - roi_h / 2.0) / scale])

            full = design(xs, ys)
            step = n_samples // _MAX_FIT_SAMPLES + 1
            fit_d, fit_s = full[::step], samples[::step]

            coef, inliers = _robust_fit(fit_d[:, :1], fit_s, unit)
            model_cols, method = 1, 'flat'
            complexity = _residual_std(fit_d[:, :1], fit_s, coef)
            if self.model != 'flat' and len(fit_s) >= _MIN_PLANE_SAMPLES:
                p_coef, p_inl = _robust_fit(fit_d, fit_s, unit)
                p_complexity = _residual_std(fit_d, fit_s, p_coef)
                if self.model == 'plane' or p_complexity < _PLANE_GAIN * complexity:
                    coef, inliers, complexity = p_coef, p_inl, p_complexity
                    model_cols, method = 3, 'plane'
            complexity *= 255.0 / max_val
            is_complex = complexity > self.max_complexity

            if not is_complex:
                pred = design(tx, ty)[:, :model_cols] @ coef
                if self.add_noise:
                    resid = fit_s[inliers] - fit_d[inliers][:, :model_cols] @ coef
                    sigma = _MAD_TO_SIGMA * np.median(np.abs(resid - np.median(resid, axis=0)), axis=0)
                    pred = pred + self._rng.normal(size=pred.shape) * sigma
        else:
            samples = None
            is_complex = True

        if pred is None:  # complex background or no background pixels at all
            if self.on_complex == 'raise':
                raise ComplexBackgroundError(
                    f'region {label} at (x, y, w, h)={bbox}: background is not simple '
                    f'(complexity {complexity:.1f} > {self.max_complexity}, ring pixels: {n_samples})')
            if self.on_complex == 'skip':
                return ComponentReport(label, bbox, 'skipped', complexity, True, n_samples)
            if self.on_complex == 'nearest' and n_samples:
                method = 'nearest'
                pred = self._nearest(ring_mask, ys, xs, samples, ty, tx, channels)
            else:
                method = 'ns' if self.on_complex == 'ns' else 'telea'
                pred = self._cv_inpaint(src, core, method, channels)[ty, tx].astype(np.float64)

        new = weight * pred + (1.0 - weight) * dst[ty, tx].astype(np.float64)
        dst[ty, tx] = np.clip(np.rint(new), 0, max_val).astype(dst.dtype)
        return ComponentReport(label, bbox, method, complexity, is_complex, n_samples)

    @staticmethod
    def _nearest(ring_mask: _NDArray, ys: _NDArray, xs: _NDArray, samples: _NDArray,
                 ty: _NDArray, tx: _NDArray, channels: int) -> _NDArray:
        """Colour of the nearest ring pixel for each target pixel."""
        src = np.ones(ring_mask.shape, dtype=np.uint8)
        src[ring_mask] = 0
        _, lab = cv.distanceTransformWithLabels(src, cv.DIST_L2, cv.DIST_MASK_5, labelType=cv.DIST_LABEL_PIXEL)
        table = np.zeros((int(lab.max()) + 1, channels), dtype=np.float64)
        table[lab[ys, xs]] = samples
        return table[lab[ty, tx]]

    @staticmethod
    def _cv_inpaint(src: _NDArray, core: _NDArray, method: str, channels: int) -> _NDArray:
        result = src.copy()
        for c in range(channels):
            result[:, :, c] = cv.inpaint(np.ascontiguousarray(src[:, :, c]), core, 3, _INPAINT_FLAGS[method])
        return result

    def _warn(self, reports: list[ComponentReport]) -> None:
        complex_count = sum(r.is_complex for r in reports)
        if complex_count:
            warnings.warn(
                f'{complex_count} of {len(reports)} regions have a complex background '
                f'(complexity > {self.max_complexity}); on_complex={self.on_complex!r} was applied '
                f'and the result may look imperfect.',
                ComplexBackgroundWarning, stacklevel=3)
