
# simple-image-inpainter

![US](https://kvaytg.ru/common/flags/us-21x16.svg) **English** | [![RU](https://kvaytg.ru/common/flags/ru-21x16.svg) Русский](README.ru.md)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue)

An OpenCV-based tool for area inpainting.

## 📚 Usage
```python
import cv2 as cv
from simple_image_inpainter import SimpleImageInpainter

# Initialize (dilation_size determines the background sampling radius)
inpainter = SimpleImageInpainter(dilation_size=5)

# Load image and mask (the mask must be grayscale)
img = cv.imread('image.jpg')
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)

# Perform inpainting
if img is not None and mask is not None:
    result = inpainter.inpaint(img, mask)
    cv.imwrite('result.jpg', result)
```

## ⚙️ Installation
```bash
pip install git+https://github.com/KvaytG/simple-image-inpainter.git
```

## 📜 License
Licensed under the **[PolyForm Noncommercial](LICENSE.md)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
