import cv2
import numpy as np
from document_recognition import (
    detect_text,
    predict_orient,
    predict_type,
    recognize_text,
    rotate_image_based_on_orientation,
)
from utils import (
    preprocess,
)


def predict_image(image_bytes: bytes) -> tuple:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess(image)
    doc_type, page_number, confidence = predict_type(preprocessed_image)
    orient_key = predict_orient(preprocessed_image)

    rotated_image = rotate_image_based_on_orientation(image, orient_key, doc_type)
    text_crop_image = detect_text(rotated_image)
    if text_crop_image is None:
        return doc_type, page_number, confidence, "unknown", "unknown"
    series, number = recognize_text(
        text_crop_image,
        remove_letters=(doc_type not in ["vehicle_passport", "vehicle_certificate"]),
    )
    return doc_type, page_number, confidence, series, number
