from Faster_R_CNN import Faster_R_CNN
from train import train


def main():
    in_channels = 3
    num_classes = 80
    data_source = '/Users/ahmet/Desktop/python/PyTorch/Faster R-CNN'
    batch_size = 1
    epochs = 10
    
    model = Faster_R_CNN(in_channels,num_classes)
    train(model,data_source,batch_size=batch_size, epochs=epochs)
    
    
if __name__ == '__main__':
    main()
