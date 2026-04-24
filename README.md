# Snow Pole Detection

Mini-project for **TDT4265: Computer Vision and Deep Learning** by Kristoffer Sandersen Nyrnes and Stener Thoresen Nordnes.

This project focused on detecting snow poles in winter road images from Trondheim using object detection models trained on the Poles2025 dataset.

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
