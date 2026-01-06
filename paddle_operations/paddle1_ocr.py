import paddle
from paddleocr import PaddleOCR
import time
from preprocessing.preprocess import PaddlePreprocessor

# Check for GPU
# print(paddle.device.is_compiled_with_cuda())

# Initialize OCR Engine Globally (Load once, use many times)
japanese_ocr = PaddleOCR(
    use_gpu=True,
    use_angle_cls=True,
    lang='japan',
    det_limit_side_len=1280,
    det_db_unclip_ratio=2.0,
    use_dilation=True,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    ocr_version="PP-OCRv4",
    show_log=False # Cleaner API logs
)

def extract_from_doc(image_input) -> list:
    """
    Args:
        image_input: Can be a file path (str) or numpy array (image)
    Returns:
        [extract_results, time_taken, processed_img]
    """
    start = time.time()
    
    # Preprocessing
    processor = PaddlePreprocessor(image_input)
    processed_img = processor.process()
    
    # OCR Operation
    # cls=True enables orientation classification (fixes rotated images)
    extract_results = japanese_ocr.ocr(processed_img, cls=True)
    
    end = time.time()
    time_taken = end - start
    
    return [extract_results, time_taken, processed_img]