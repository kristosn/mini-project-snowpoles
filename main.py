from run_model import run_yolo, run_rf_detr
from pathlib import Path
from tune_hyperparams import tune_hyperparams

def main():
    
    IDUN = Path("/cluster").exists()
    print(f"Using IDUN: {IDUN}")
    
    epochs = 300
    imgsz = 640
    resolution = 512
    
    model_path_yolo = "yolo26s.pt"
    model_name_rfdetr = "RFDETRSmall"
    
    #iterations = 50
    #tune_hyperparams("v1", model_path_yolo, epochs, iterations)
    #tune_hyperparams("iphone", model_path_yolo, epochs, iterations)
    
    # run_yolo("v1", model_path_yolo, imgsz, epochs)
    # run_yolo("iphone", model_path_yolo, imgsz, epochs)
    
    run_rf_detr("v1", model_name_rfdetr, epochs, resolution)
    run_rf_detr("iphone", model_name_rfdetr, epochs, resolution)
    
if __name__ == "__main__":
    main()
    