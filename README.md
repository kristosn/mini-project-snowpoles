# Snow Pole Detection

Mini-project for **TDT4265: Computer Vision and Deep Learning** by Kristoffer Sandersen Nyrnes and Stener Thoresen Nordnes.

We have train and evaluate object detection models for detecting snow poles in winter road images from Trondheim using the **Poles2025** dataset.

## Models

- YOLO26s
- RF-DETR Small

## Dataset

- Poles2025
- 1 class: `pole`
- YOLO annotations converted to COCO format for RF-DETR

## Run

```bash
python main.py
