# Snow Pole Detection

**Authors:** Kristoffer Sandersen Nyrnes, Stener Thoresen Nordnes

Mini-project for **TDT4265: Computer Vision and Deep Learning**.

This project focuses on **real-time object detection of snow poles** in road images from Trondheim.  
The goal is to detect snow poles in challenging winter conditions, where they can help indicate the road boundaries.

## Project goal

The task is to train and evaluate object detection models for detecting **snow poles** in images from the **Poles2025** dataset.

We worked with:
- **1 object class**: `pole`
- object detection on two datasets:
  - `roadpoles_v1`
  - `Road_poles_iPhone`

## Dataset

Dataset used:
- **Poles2025**

The dataset contains annotations in **YOLO format**.  
For some parts of the project, these annotations were also converted to **COCO format** for compatibility with other models.

> Note: The dataset is not redistributed in this repository.

## Models

We used the following models:
- **YOLO**
- **RF-DETR**

These models were trained and evaluated on the snow pole detection task.

## Project structure

Example file overview:

- `main.py` – runs the full pipeline
- `run_model.py` – handles training and prediction for selected models
- `train.py` – training functions for YOLO and RF-DETR
- `predict.py` – prediction functions
- `yolo_to_coco.py` – converts YOLO annotations to COCO format

## How to run

Run the full project with:

```bash
python main.py
