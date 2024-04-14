import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO

type_model = (
    torch.jit.load("./weights/type_classifier.torchscript", map_location="cuda:0")
    .half()
    .eval()
)
orient_model = (
    torch.jit.load("./weights/orient_classifier.torchscript", map_location="cuda:0")
    .half()
    .eval()
)
text_detection_model = YOLO("./weights/text_detector.pt")
text_recognition_model = PaddleOCR(use_angle_cls=True, lang="en")
