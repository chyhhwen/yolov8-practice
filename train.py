import multiprocessing

import ultralytics
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO('yolov8s.pt')
    model.train(
        data="data.yaml",
                mode="detect",
                epochs=100,
                imgsz=640,
                device=0
    )