from torch.utils.data import DataLoader
from COCO import COCODataset
import torch.nn as nn
from tqdm import tqdm
import torch


def train(model: nn.Module, batch_size: int = 1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    model.to(device)
    
    train = COCODataset('/Users/ahmet/Desktop/python/PyTorch/Faster R-CNN')
    train_dl = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    
    test = COCODataset('/Users/ahmet/Desktop/python/PyTorch/Faster R-CNN', 'test')
    test_dl = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    
