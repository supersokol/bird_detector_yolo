# test task - Bird Detector YOLO26 

## Dataset

bird detection Computer Vision Dataset (https://www.universe.roboflow.com/ryu-myoungwoo/bird-detection-zqcdr)
*you can find additional info in Jupytre notebook (bird-detector.ipynb)

## Quickstart
### 1) Create venv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
### 2) Install dependencies

```bash
pip install -r requirements.txt 
```
### 3) Obtain training dataset (bird detection Computer Vision Dataset)

open URL https://universe.roboflow.com/ryu-myoungwoo/bird-detection-zqcdr/dataset/5/download/yolo26 use your google or other account to auth into roboflow chose "Download dataset" you'll get file bird detection.v5i.yolo26.zip unpack it into project datasets\bird detection.v5i.yolo26\ directory so the data.yaml will be available at "datasets\bird detection.v5i.yolo26\data.yaml"
### 4) Train and validate YOLO26

```bash
python train.py
```

#### exported models weigths 
when we are using model.train() by default best weights of training sequence are saved to "runs\detect\train\weights\best.pt"

### 5) Run Prediction

to test on video:
```bash
python predict_video.py -i examples/input/video.mp4 -o examples/output/video_test.mp4
```
to test on images:
```bash
python predict_img.py
```

## additional info
you can find additional info in Jupytre notebook (bird-detector.ipynb)
