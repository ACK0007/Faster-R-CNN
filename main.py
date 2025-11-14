from Faster_R_CNN import Faster_R_CNN
from evaluation import evaluate
from train import train


def main():
    in_channels = 3
    num_classes = 80
    data_source = '/Users/ahmet/Desktop/python/PyTorch/COCO'
    batch_size = 1
    epochs = 10
    
    model = Faster_R_CNN(in_channels, num_classes)
    train(model, data_source, batch_size=batch_size, epochs=epochs)
    evaluate(model, data_source)
    
    
if __name__ == '__main__':
    main()
