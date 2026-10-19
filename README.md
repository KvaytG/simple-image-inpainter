
# simple-image-inpainter

**English** | [Русский](README.ru.md)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue)

An OpenCV-based tool for inpainting regions on **simple backgrounds**: flat colour, smooth gradients, light noise or grain. It is not meant for complex textures; such regions are detected and reported instead of being filled blindly.

## 📚 Usage
```python
import cv2 as cv
from simple_image_inpainter import SimpleImageInpainter

# dilation_size: radius (px) by which every mask region is grown to cover halos
inpainter = SimpleImageInpainter(dilation_size=5)

img = cv.imread('image.jpg')                              # uint8/uint16; gray, BGR or BGRA
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)         # values > 127 are masked

if img is not None and mask is not None:
    result = inpainter.inpaint(img, mask)
    cv.imwrite('result.jpg', result)
```

Per-region diagnostics:
```python
result, reports = inpainter.inpaint_with_report(img, mask)
for r in reports:
    print(r.bbox, r.method, round(r.complexity, 1), r.is_complex)
```

## 🧠 How it works
1. Every connected mask region is grown by `dilation_size` px.
2. The background is sampled from a ring of `ring_width` px *outside* that area (pixels of other mask regions are excluded).
3. A robust model is fitted to the ring: a flat colour or a linear gradient (`model='auto'` picks the plane only when it clearly fits better). Outliers are rejected.
4. The area is refilled from the model, with grain matching the ring noise, and the edge is feathered.
5. If the ring is not a simple background (residual std above `max_complexity`), the region is handled by `on_complex` and a `ComplexBackgroundWarning` is emitted.

## ⚙️ Parameters
| Parameter        | Default     | Meaning                                                                     |
|------------------|-------------|-----------------------------------------------------------------------------|
| `dilation_size`  | `3`         | Radius (px) the mask regions are grown by                                   |
| `ring_width`     | `4`         | Width (px) of the background sampling ring                                  |
| `feather`        | `1.0`       | Gaussian sigma (px) of the soft edge; `0` = hard edge                       |
| `model`          | `'auto'`    | `'flat'`, `'plane'` or `'auto'`                                             |
| `max_complexity` | `15.0`      | Ring residual std (0..255 scale) above which a background counts as complex |
| `on_complex`     | `'nearest'` | `'nearest'`, `'telea'`, `'ns'`, `'skip'` or `'raise'`                       |
| `add_noise`      | `True`      | Add grain matching the ring noise                                           |
| `mask_threshold` | `127`       | Mask values above this are masked (bool masks are used as is)               |
| `seed`           | `None`      | Seed for reproducible grain                                                 |

`'nearest'` copies the colour of the nearest ring pixel, which suits piecewise-flat backgrounds (e.g. an edge between two panels). Typical complexity values: 1-5 for clean or noisy simple backgrounds, tens for edges and textures.

## ⚙️ Installation
```bash
pip install git+https://github.com/KvaytG/simple-image-inpainter.git
```
Development: `pip install -e .[dev] && pytest`

## 📜 License
Licensed under the **[PolyForm Noncommercial](LICENSE.md)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
