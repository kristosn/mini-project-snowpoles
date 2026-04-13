from ultralytics import YOLO
from time import perf_counter
from pathlib import Path

def tune_hyperparams(dataset, model_path, epochs, iterations):
    if dataset == "iphone":
        data = "Poles2025/Road_poles_iPhone/data.yaml"
    elif dataset == "v1":
        data= "Poles2025/roadpoles_v1/data.yaml"      
        
    model = YOLO(model_path)
    start = perf_counter()
    model.tune(
        data=data,
        epochs=epochs, 
        iterations=iterations,
        name=f"tune_{dataset}"
    )
    end = perf_counter()
    time_elapsed_tune = end - start
    IDUN = Path("/cluster").exists()
    with open("time_tune_hyperparams.txt", "a") as f:
        f.write(f"Dataset: {dataset}, model: {model_path}, tuning time (s): {time_elapsed_tune:.2f}, IDUN: {IDUN}\n")
    