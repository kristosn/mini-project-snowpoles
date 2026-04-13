from ultralytics import YOLO
from time import perf_counter
from pathlib import Path
from rfdetr import RFDETRNano, RFDETRSmall

def train_yolo(dataset, model_path, epochs, imgsz, project):
    print("Starting training with YOLO...")       
    if dataset == "iphone":
        data = "Poles2025/Road_poles_iPhone/data.yaml"
    elif dataset == "v1":
        data= "Poles2025/roadpoles_v1/data.yaml"       
        
    model = YOLO(model_path)
    start = perf_counter()
    results = model.train(
        data=data, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=-1, 
        name=f"train_yolo_{dataset}_{imgsz}",
        project=project,
        patience=50
    )
    end = perf_counter()
    time_elapsed = end - start
    return results, time_elapsed

def train_rf_detr(dataset, model_name, epochs, resolution, output_dir):
    print("Starting training with RF-DETR")
    if dataset == "iphone":
        data = "Poles2025/Road_poles_iPhone"
    elif dataset == "v1":
        data = "Poles2025/roadpoles_v1"
        
    if model_name.lower() == "rfdetrnano":
        model = RFDETRNano(resolution=resolution)
    elif model_name.lower() == "rfdetrsmall":
        model = RFDETRSmall(resolution=resolution)
    start = perf_counter()
    results = model.train(
        dataset_dir=data,
        epochs=epochs,
        batch_size=2,
        grad_accum_steps=8,
        output_dir=output_dir,
        early_stopping=True,
        early_stopping_patience=50
    )
    end = perf_counter()
    time_elapsed = end - start
    return results, time_elapsed