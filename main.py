from run_model import run_yolo
from pathlib import Path
import subprocess, sys

def main():
    
    IDUN = Path("/cluster").exists()
    print(f"Using IDUN: {IDUN}")
    
    epochs = 300
    imgsz = 640
    resolution = 512
    
    model_path_yolo = "yolo26s.pt"
    model_name_rfdetr = "RFDETRSmall"
    
    if IDUN:
        subprocess.run([sys.executable, "run_model.py", "v1", model_name_rfdetr, str(epochs), str(resolution)], check=True)
        subprocess.run([sys.executable, "run_model.py", "iphone", model_name_rfdetr, str(epochs), str(resolution)], check=True)
    else:
        from run_model import run_rf_detr
        run_rf_detr("v1", model_name_rfdetr, epochs, resolution)
        run_rf_detr("iphone", model_name_rfdetr, epochs, resolution)
    
    run_yolo("v1", model_path_yolo, imgsz, epochs)
    run_yolo("iphone", model_path_yolo, imgsz, epochs)
    
if __name__ == "__main__":
    main()