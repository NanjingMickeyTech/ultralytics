from ultralytics import YOLO
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())


if __name__ == '__main__':
    model = YOLO("/home/wqq/ultralytics/yolov8s.yaml")
    results = model.train(data="/home/wqq/ultralytics/coco_fishnet.yaml", epochs=100, imgsz=640)