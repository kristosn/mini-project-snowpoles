from ultralytics import YOLO, RTDETR
from time import perf_counter

def train_yolo(dataset, model, epochs, imgsz, IDUN):
    if dataset == "iphone":
        data = "data_idun_iphone.yaml" if IDUN else "data_local_iphone.yaml"
    elif dataset == "v1":
        data = "data_idun_v1.yaml" if IDUN else "data_local_v1.yaml"
    model = YOLO(model)
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
    
def train_rt_detr(dataset, imgsz, epochs, IDUN):
    if dataset == "iphone":
        data = "data_idun_iphone.yaml" if IDUN else "data_local_iphone.yaml"
    elif dataset == "v1":
        data = "data_idun_v1.yaml" if IDUN else "data_local_v1.yaml"
    model = RTDETR("rtdetr-l.pt")
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