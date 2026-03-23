from ultralytics import YOLO, RTDETR
from time import perf_counter

def train_yolo(IDUN, epochs, imgsz):
    data = "data_idun.yaml" if IDUN else "data_local.yaml"
    model = YOLO("yolo11n.pt")
    start = perf_counter()
    results = model.train(
        data=data, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=-1, 
        name=f"train_yolo_{imgsz}"
    )
    end = perf_counter()
    time_elapsed = end - start
    return results, time_elapsed
    
def train_rt_detr(IDUN, epochs, imgsz):
    data = "data_idun.yaml" if IDUN else "data_local.yaml"
    model = RTDETR("rtdetr-l.pt")
    # batch = 2 if imgsz >= 1920 else -1
    start = perf_counter()
    results = model.train(
        data=data, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=-1, 
        name=f"train_rt_detr_{imgsz}",
        )
    end = perf_counter()
    time_elapsed = end - start
    return results, time_elapsed