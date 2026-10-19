
# simple-image-inpainter

[English](README.md) | **Русский**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue)

Инструмент на основе OpenCV для заливки областей на **простых фонах**: однотонных, с плавным градиентом, лёгким шумом или зерном. Для сложных текстур он не предназначен: такие области не заливаются вслепую, а обнаруживаются и попадают в отчёт.

## 📚 Использование
```python
import cv2 as cv
from simple_image_inpainter import SimpleImageInpainter

# dilation_size — радиус (px), на который расширяется каждая область маски (закрывает ореолы)
inpainter = SimpleImageInpainter(dilation_size=5)

img = cv.imread('image.jpg')                              # uint8/uint16; серое, BGR или BGRA
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)         # значения > 127 считаются маской

if img is not None and mask is not None:
    result = inpainter.inpaint(img, mask)
    cv.imwrite('result.jpg', result)
```

Диагностика по областям:
```python
result, reports = inpainter.inpaint_with_report(img, mask)
for r in reports:
    print(r.bbox, r.method, round(r.complexity, 1), r.is_complex)
```

## 🧠 Как это работает
1. Каждая связная область маски расширяется на `dilation_size` px.
2. Фон сэмплируется из кольца шириной `ring_width` px *снаружи* этой области (пиксели других областей маски исключаются).
3. По кольцу устойчиво подбирается модель: плоский цвет или линейный градиент (`model='auto'` выбирает плоскость, только если она заметно точнее). Выбросы отбрасываются.
4. Область перезаполняется по модели с зерном, соответствующим шуму кольца, а край растушёвывается.
5. Если кольцо не является простым фоном (разброс остатков выше `max_complexity`), область обрабатывается по `on_complex` и выдаётся `ComplexBackgroundWarning`.

## ⚙️ Параметры
| Параметр         | По умолчанию | Значение                                                                    |
|------------------|--------------|-----------------------------------------------------------------------------|
| `dilation_size`  | `3`          | Радиус (px) расширения областей маски                                       |
| `ring_width`     | `4`          | Ширина (px) кольца для выборки фона                                         |
| `feather`        | `1.0`        | Сигма (px) мягкого края; `0` — жёсткий край                                 |
| `model`          | `'auto'`     | `'flat'`, `'plane'` или `'auto'`                                            |
| `max_complexity` | `15.0`       | Разброс остатков кольца (шкала 0..255), выше которого фон считается сложным |
| `on_complex`     | `'nearest'`  | `'nearest'`, `'telea'`, `'ns'`, `'skip'` или `'raise'`                      |
| `add_noise`      | `True`       | Добавлять зерно, подобное шуму кольца                                       |
| `mask_threshold` | `127`        | Значения маски выше порога считаются маской (булевы маски берутся как есть) |
| `seed`           | `None`       | Seed для воспроизводимого зерна                                             |

`'nearest'` копирует цвет ближайшего пикселя кольца и подходит для фонов из плоских кусков (например, граница двух панелей). Типичные значения сложности: 1–5 для чистых и шумных простых фонов, десятки для границ и текстур.

## ⚙️ Установка
```bash
pip install git+https://github.com/KvaytG/simple-image-inpainter.git
```
Разработка: `pip install -e .[dev] && pytest`

## 📜 Лицензия
Распространяется по лицензии **[PolyForm Noncommercial](LICENSE.md)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
