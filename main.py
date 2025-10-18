from Faster_R_CNN import Faster_R_CNN
from train import train
import torch.nn as nn
import torch


def main():
    model = Faster_R_CNN(3,80,9)
    train(model)
    
if __name__ == 'main':
    main()
