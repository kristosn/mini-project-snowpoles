from ultralytics import YOLO, RTDETR
from train import train_yolo, train_rt_detr
from predict import predict
from pathlib import Path

def write_to_file(model, imgsz, time_elapsed_train, time_elapsed_predict, file):
    with open(file, "a") as f:
        f.write(f"Model: {model}, imgsz: {imgsz}, training time (s): {time_elapsed_train:.2f}, prediction time (s): {time_elapsed_predict:.2f}\n")
    print(f"Written to file {file}.")
    
def main():
    
    IDUN = True
   
    PATH_TO_TEST_SET = (
        "/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/test/images"
        if IDUN else
        "Poles2025/roadpoles_v1/test/images"
    )
    
    epochs = 200
    img_sizes = (640, 1280, 1920)
    
    for imgsz in img_sizes:
        
        results_yolo, time_elapsed_train_yolo = train_yolo(IDUN, epochs, imgsz)
        best_model_yolo = Path(results_yolo.save_dir) / "weights/best.pt"
        best_model_yolo = YOLO(best_model_yolo)
        time_elapsed_predict_yolo = predict(
            model=best_model_yolo,
            project="runs",
            source=PATH_TO_TEST_SET,
            name=f"predict_yolo_{imgsz}"
        )
        
        write_to_file("YOLO11n", imgsz, time_elapsed_train_yolo, time_elapsed_predict_yolo, "time.txt")

    for imgsz in img_sizes:
        
        results_rt_detr, time_elapsed_train_rt_detr = train_rt_detr(IDUN, epochs, imgsz)
        best_model_rt_detr = Path(results_rt_detr.save_dir) / "weights/best.pt"
        best_model_rt_detr = RTDETR(best_model_rt_detr)
        time_elapsed_predict_rt_detr = predict(
            model=best_model_rt_detr,
            project="runs",
            source=PATH_TO_TEST_SET, 
            name=f"predict_rt_detr_{imgsz}"
        )
        
        write_to_file("RT-DETR", imgsz, time_elapsed_train_rt_detr, time_elapsed_predict_rt_detr, "time.txt")
        
    
    
if __name__ == "__main__":
    main()
    