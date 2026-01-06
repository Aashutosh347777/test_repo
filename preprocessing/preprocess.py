import cv2
import numpy as np

class PaddlePreprocessor:
    def __init__(self, image_input):
        # Handle string path
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input)
            if self.image is None:
                raise ValueError(f"Image not found at path: {image_input}")
        # Handle numpy array (from API)
        elif isinstance(image_input, np.ndarray):
            self.image = image_input
        else:
            raise ValueError("Invalid input type. Expected file path or numpy array.")

    def add_padding(self, image, pad_size=20):
        return cv2.copyMakeBorder(
            image, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    def optimize_resolution(self, image, min_width=1000):
        h, w = image.shape[:2]
        if w < min_width:
            scale_factor = min_width / w
            return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        return image

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def simple_denoise(self, image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    def process(self):
        img = self.image.copy()
        img = self.optimize_resolution(img)
        img = self.enhance_contrast(img)
        img = self.simple_denoise(img)
        img = self.add_padding(img)
        return img