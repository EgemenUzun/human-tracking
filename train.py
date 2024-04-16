from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Ensure multiprocessing is properly initialized on Windows
    model = YOLO('yolov8m.pt')
    results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=0)
