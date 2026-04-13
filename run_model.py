from train import train_yolo, train_rf_detr
from predict import predict_yolo, predict_rf_detr
from pathlib import Path

def write_to_file(dataset, model, imgsz, time_elapsed_train, time_elapsed_predict, file):
    IDUN = Path("/cluster").exists()
    with open(file, "a") as f:
        f.write(f"Dataset: {dataset}, Model: {model}, imgsz: {imgsz}, training time (s): {time_elapsed_train:.2f}, prediction time: {time_elapsed_predict:.2f}, IDUN: {IDUN}\n")

def run_yolo(dataset, model_path, imgsz, epochs):
    project = f"runs_{model_path.replace('.pt', '')}"
    if dataset == "v1":
        PATH_TO_TEST_SET = "Poles2025/roadpoles_v1/test/images"
    elif dataset == "iphone":
        PATH_TO_TEST_SET = "Poles2025/Road_poles_iPhone/test/images"
        
    results, time_elapsed_train_yolo = train_yolo(dataset, model_path, epochs, imgsz, project)
    best_model_path = Path(results.save_dir) / "weights/best.pt"
    time_elapsed_predict = predict_yolo(best_model_path, PATH_TO_TEST_SET, project, f"predict_yolo_{dataset}_{imgsz}", imgsz)
    write_to_file(dataset, model_path.replace('.pt', ''), imgsz, time_elapsed_train_yolo, time_elapsed_predict, "time.txt")

def run_rf_detr(dataset, model_name, epochs, resolution):
    output_dir = f"runs_{model_name}_{dataset}"
    if dataset == "v1":
        PATH_TO_TEST_SET = "Poles2025/roadpoles_v1/test/images"
    elif dataset == "iphone":
        PATH_TO_TEST_SET = "Poles2025/Road_poles_iPhone/test/images"
        
    results, time_elapsed_train = train_rf_detr(dataset, model_name, epochs, resolution, output_dir)
    best_model_path = Path(output_dir) / "checkpoint_best_total.pth"
    time_elapsed_predict = predict_rf_detr(best_model_path, model_name, dataset, PATH_TO_TEST_SET, resolution, threshold=0.5)
    write_to_file(dataset, model_name, resolution, time_elapsed_train, time_elapsed_predict, "time.txt")