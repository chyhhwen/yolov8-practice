import multiprocessing

import ultralytics
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO('yolov8s.pt')
    model.train(
        data="data.yaml",
                mode="detect",
                epochs=1000,
                imgsz=640,
                batch=100,
                patience=500,
                device=0
    )