import cv2
import numpy as np

def draw_box(paddle_out, img: np.ndarray):
    """
    Draws bounding boxes on the image and returns the image array.
    """
    output_img = img.copy()
    
    # Config
    thickness = 2
    color = (0, 255, 0)
    text_color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1

    if not paddle_out or paddle_out[0] is None:
        return output_img

    for items in paddle_out[0]:
        box = items[0]
        text = items[1][0]
        confi = str(round(items[1][1], 2))
        
        xs = [int(pt[0]) for pt in box]
        ys = [int(pt[1]) for pt in box]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Label
        label_position = (x_min, y_min - 5 if y_min > 5 else y_min + 5)
        # Using simple text for now. For Japanese characters in CV2, you need PIL/Pillow.
        # We stick to displaying confidence score to avoid encoding issues in OpenCV.
        cv2.putText(output_img, confi, label_position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)
            
    return output_img