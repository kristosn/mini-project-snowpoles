from ultralytics import YOLO, RTDETR
from train import train_yolo, train_rt_detr
from predict import predict
from run_models import run_yolo, run_rt_detr
from pathlib import Path

    
def main():
    
    IDUN = True
    SAHI = False
    epochs = 300
    img_sizes = (640, 1280)
    yolo_models = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
    
    run_yolo("v1", img_sizes, yolo_models, epochs, IDUN)
    run_yolo("iphone", img_sizes, yolo_models, epochs, IDUN)
    
    run_rt_detr("v1", img_sizes, epochs, IDUN)
    run_rt_detr("iphone", img_sizes, epochs, IDUN)
    
        
    
if __name__ == "__main__":
    main()
    