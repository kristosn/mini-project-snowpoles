from ultralytics import YOLO
from rfdetr import RFDETRNano, RFDETRSmall
from pathlib import Path
from PIL import Image, ImageDraw
from time import perf_counter

def predict_yolo(model_path, source, project, name, imgsz):
    print("Starting prediction with YOLO...")
    start = perf_counter()
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        project=project,
        name=name,
        imgsz=imgsz,
        save_txt=True,
        save_conf=True, # <--- This adds the probability of each predicted box
        exist_ok=True
    )
    end = perf_counter()
    time_elapsed = end - start
    return time_elapsed

def bbox_to_yolobox(x1, y1, x2, y2, image_w, image_h):
    cx = ((x1 + x2) / 2) / image_w
    cy = ((y1 + y2) / 2) / image_h
    w  = (x2 - x1) / image_w
    h  = (y2 - y1) / image_h
    return cx, cy, w, h

def draw_bounding_box(image_path, images_dir, predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i in range(len(predictions.xyxy)):
        x1, y1, x2, y2 = predictions.xyxy[i]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        class_id = predictions.class_id[i]
        conf = predictions.confidence[i]
        draw.text((x1, max(0, y1 - 10)), f"{conf:.2f}", fill="green")
    image.save(images_dir / image_path.name)
        
        
def predict_rf_detr(model, model_name, dataset, source, resolution, threshold=0.5):
    print("Starting prediction with RF-DETR...")
    
    if model_name.lower() == "rfdetrnano":
        model = RFDETRNano(pretrain_weights=str(model), resolution=resolution)
    elif model_name.lower() == "rfdetrsmall":
        model = RFDETRSmall(pretrain_weights=str(model), resolution=resolution)
    
    output_dir = Path(f"runs_{model_name}_{dataset}") / "predict"
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(Path(source).glob("*.PNG")) + list(Path(source).glob("*.jpg"))
    
    predict_times = []
    
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        start = perf_counter()
        predictions = model.predict(str(image_path), threshold=threshold)
        end = perf_counter()
        predict_times.append(end - start)
        label_lines = []
        
        for i in range(len(predictions.xyxy)):
            x1, y1, x2, y2 = predictions.xyxy[i]
            cx, cy, w, h = bbox_to_yolobox(x1, y1, x2, y2, image_width, image_height)
            class_id = int(predictions.class_id[i])
            conf = float(predictions.confidence[i])
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
        
        label_file = labels_dir / (image_path.stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))
        draw_bounding_box(image_path, images_dir, predictions)
    time_elapsed = sum(predict_times)
    return time_elapsed
            
        
    
    
    
    

