import paddle
from paddleocr import PaddleOCR
import time

print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())

img_path = r"C:\Users\LEGION\Documents\inficare\OCR\images\japanese\Mahat_neel.png"

ocr = PaddleOCR(use_angle_cls = True, lang = "japan")

start = time.time()
result = ocr.ocr(img_path)
end = time.time()
print(type(result))
print(result)

print(f"\n Total time take -> {end-start}")

# displaying the bounding boxes for the output
for 