from ultralytics import YOLO
from time import perf_counter

def predict(model, source, project, name):
    start = perf_counter()
    model.predict(
        source=source,
        project=project,
        name=name,
        save_txt=True,
        save_conf=True # <--- This adds the probability of each predicted box
    )
    end = perf_counter()
    time_elapsed = end - start
    return time_elapsed
        