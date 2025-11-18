from Faster_R_CNN import Faster_R_CNN
from evaluation import evaluate
from train import train
import argparse


def main(args: argparse.Namespace):
    in_channels = 3
    num_classes = 80
    data_source = args.data_source
    batch_size = args.batch_size
    epochs = args.epochs
    
    model = Faster_R_CNN(in_channels, num_classes)
    train(model, data_source, batch_size=batch_size, epochs=epochs, lr=args.lr)
    evaluate(model, data_source)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_source", type=str, default='/Users/ahmet/Desktop/python/PyTorch/COCO')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-epochs", type=float, default=10)
    args = parser.parse_args()
    main(args)
