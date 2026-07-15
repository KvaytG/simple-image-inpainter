
# simple-image-inpainter

[![US](https://kvaytg.ru/common/flags/us-21x16.svg) English](README.md) | ![RU](https://kvaytg.ru/common/flags/ru-21x16.svg) **Русский**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue)

Инструмент на основе OpenCV для заливки областей.

## 📚 Использование
```python
import cv2 as cv
from simple_image_inpainter import SimpleImageInpainter

# Инициализация (dilation_size — радиус захвата фона вокруг области)
inpainter = SimpleImageInpainter(dilation_size=5)

# Загрузка изображения и маски (маска должна быть ЧБ)
img = cv.imread('image.jpg')
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)

# Заливка областей
if img is not None and mask is not None:
    result = inpainter.inpaint(img, mask)
    cv.imwrite('result.jpg', result)
```

## ⚙️ Установка
```bash
pip install git+https://github.com/KvaytG/simple-image-inpainter.git
```

## 📜 Лицензия
Распространяется по лицензии **[PolyForm Noncommercial](LICENSE.md)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
