import cv2 as cv
import numpy as np
from typing import Optional, Union

_NDArray = np.ndarray


class SimpleImageInpainter:
    def __init__(self, dilation_size: int = 3) -> None:
        self.kernel: _NDArray = np.ones((dilation_size, dilation_size), np.uint8)

    def _get_border_mask(self, component_mask: _NDArray) -> tuple[_NDArray, _NDArray]:
        h, w = component_mask.shape[:2]
        dilated: _NDArray = cv.dilate(component_mask, self.kernel, iterations=1)
        border_mask: _NDArray = cv.subtract(dilated, component_mask)
        border_mask[0:1, :] = 0
        border_mask[h - 1:h, :] = 0
        border_mask[:, 0:1] = 0
        border_mask[:, w - 1:w] = 0
        return dilated, border_mask

    @staticmethod
    def _get_dominant_color(image: _NDArray, border_mask: _NDArray) -> Optional[Union[int, list[int]]]:
        bg_pixels: _NDArray = image[border_mask > 0]
        if bg_pixels.size == 0:
            return None
        if len(image.shape) == 3:
            bg_pixels_2d: _NDArray = bg_pixels.reshape(-1, 3)
            colors, counts = np.unique(bg_pixels_2d, axis=0, return_counts=True)
            fill_color: _NDArray = colors[counts.argmax()]
            return [int(c) for c in fill_color]
        else:
            colors, counts = np.unique(bg_pixels, return_counts=True)
            return int(colors[counts.argmax()])

    def inpaint(self, image: _NDArray, mask: _NDArray) -> _NDArray:
        img_result: _NDArray = image.copy()
        num_labels: int
        labels: _NDArray
        num_labels, labels = cv.connectedComponents(mask)
        for label in range(1, num_labels):
            component_mask: _NDArray = np.uint8(labels == label) * 255
            dilated_mask, border_mask = self._get_border_mask(component_mask)
            fill_color = self._get_dominant_color(image, border_mask)
            if fill_color is not None:
                img_result[dilated_mask > 0] = fill_color
        return img_result
