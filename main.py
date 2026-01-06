from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
import cv2
import numpy as np
import base64
import json

from paddle_operations.paddle1_ocr import extract_from_doc
from paddle_operations.draw_bounding_box import draw_box
from parsers import master_parser

app = FastAPI(title="PaddleOCR Extraction API")

def image_to_base64(image_array):
    """Convert numpy image to base64 string for JSON response"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/extract")
async def extract_text(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 1. Read Image File
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            results.append({"filename": file.filename, "error": "Invalid image file"})
            continue

        # 2. Run OCR Pipeline
        # extract_from_doc returns [results, time, processed_img]
        ocr_out = extract_from_doc(img)
        raw_results = ocr_out[0]
        time_taken = ocr_out[1]
        processed_img = ocr_out[2]

        # 3. Parse Data (Regex)
        parsed_data = master_parser(raw_results)

        # 4. Draw Bounding Boxes
        visualized_img = draw_box(raw_results, processed_img)
        img_base64 = image_to_base64(visualized_img)

        # 5. Structure Response
        results.append({
            "filename": file.filename,
            "processing_time_seconds": round(time_taken, 4),
            "extracted_data": parsed_data,
            "image_with_boxes": img_base64  # Frontend can display this
        })

    return {"status": "success", "processed_count": len(results), "data": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)