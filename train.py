from utils import setup_logger
import os
import sys
import argparse
from typing import Dict
from datetime import datetime
from ultralytics import YOLO

log = setup_logger()

def train_model(
    model: YOLO, data_path: str = "datasets/bird detection.v5i.yolo26/data.yaml", tile: bool = True, 
    epochs: int = 50, warmup_epochs: float = 1.0, 
    box: float = 10.0, 
    dropout: float = 0.2, weight_decay: float = 0.001, 
    imgsz: int = 640, 
    patience: int = 10, batch: int = -1, 
    scale: float = 0.7, degrees: int = 15, fliplr: float = 0.5, perspective: float = 0.0005,
    shear: float = 5.0, mosaic: int = 0.5, close_mosaic: int = 10, cutmix: float = 0.1, mixup: int = 0.0, copy_paste: int = 0.0, optimizer: str = "AdamW", seed: int = 647, lr0: float = 0.001, lrf: float = 0.01, cos_lr: bool = True, freeze: int = 23, 
    device: str = "cpu", workers: int = 4, project: str = "runs/train", name: str = "bird_detector") -> Dict:
    """
    Train the YOLO model.

    Args:
        model (YOLO): The YOLO model to train.
        data_path (str): Path to the dataset YAML file.
        tile (bool): Whether to use tiling for training. Tiling can help with small object detection by dividing images into smaller tiles.
        epochs (int): Number of training epochs.
        warmup_epochs (float): Number of warmup epochs to gradually increase the learning rate.
        box (float): Box loss gain for training.
        dropout (float): Dropout rate for regularization.
        weight_decay (float): Weight decay for regularization.
        imgsz (int): Image size for training.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        batch (int): Batch size for training. -1 means auto mode for 60% of available GPU memory
        scale (float): Random scaling augmentation range. Helps model detect objects at different sizes
        degrees (int): Maximum rotation angle for augmentation.
        fliplr (float): Probability of horizontal flip augmentation.
        perspective (float): Applies a full perspective transformation along both x-axis and y-axis, simulating how objects appear when viewed from different depths or angles. The perspective hyperparameter defines the perspective magnitude.
        shear (float): Introduces a geometric transformation that skews the image along both x-axis and y-axis. Helps models generalize to variations in viewing angles caused by slight tilts or oblique viewpoints.
        mosaic (float): Probability of using mosaic augmentation, which combines 4 images. Especially useful for small  object detection.
        close_mosaic (int): Disables mosaic in the last N epochs to stabilize training before completion.
        cutmix (float): CutMix augmentation probability. Combines image regions while maintaining local features
        mixup (float): Mixup augmentation probability.
        copy_paste (float): Copy-paste augmentation probability.
        optimizer (str): Optimizer to use for training.
        seed (int): Random seed for reproducibility. 
        lr0 (float): Initial learning rate.
        lrf (float): Final learning rate as a fraction of the initial learning rate.
        cos_lr (bool): Whether to use cosine learning rate scheduling.
        freeze (int): Number of epochs to freeze the backbone layers.
        device (int): Device index for training (0 for GPU, -1 for CPU).
        workers (int): Number of data loading workers.
        project (str): Directory to save training results.
        name (str): Name of the training run.

    * https://docs.ultralytics.com/guides/yolo26-training-recipe - says:
    "for Small datasets (< 1,000 images):
    -    Reduce augmentation strength: mosaic=0.5, mixup=0.0, copy_paste=0.0
    -    Lower learning rate with an explicit optimizer: optimizer=AdamW, lr0=0.001
    -    Use fewer epochs with patience: epochs=50, patience=20
    -    Consider freezing backbone layers: freeze=10"
    and 
    "Increase degrees if objects appear at arbitrary rotations"
    https://docs.ultralytics.com/guides/finetuning-guide - says:
    "for Very small dataset	set freeze=23	
    Only the detection head trains, minimizing overfitting risk"
    Returns:
        Dict: Training results.
    """
    return model.train( 
        data=data_path, 
        epochs = epochs, warmup_epochs = warmup_epochs, 
        box = box, 
        dropout = dropout, weight_decay = weight_decay, 
        imgsz = imgsz, 
        patience = patience, batch = batch, 
        scale = scale, degrees = degrees, fliplr = fliplr, perspective = perspective,
        shear = shear, mosaic = mosaic, close_mosaic = close_mosaic, cutmix = cutmix, mixup = mixup, copy_paste = copy_paste, 
        optimizer = optimizer, seed = seed, lr0 = lr0, lrf = lrf, cos_lr = cos_lr, freeze = freeze, 
        device = device, workers = workers, project = project, name = name, exist_ok = True)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', default=os.getenv('dataset', 'datasets/bird detection.v5i.yolo26/data.yaml'), type=str, help='path to dataset for training by default bird detection.v5i.yolo26. dataset will be used')

    args = argparser.parse_args()
    dataset = args.dataset
    log = setup_logger()  # Set up the logger
    try:
        model = YOLO("yolo26n.pt")  # load a pretrained YOLO26n model
        print(f"Training model on dataset: {dataset}")
        results = train_model(model, data_path=dataset)  #, epochs=100, imgsz=640)
        print(f"Training completed. Results: {results}")
        # Export the model
        model.export()
        print("Model exported successfully.")
        sys.exit(0)
    except Exception as err:
        log.info('Error occured!')
        log.info(' %s' % err)
        sys.exit(1)