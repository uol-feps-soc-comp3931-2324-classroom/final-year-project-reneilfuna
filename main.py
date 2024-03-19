import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import Fingers
from model import SimpleConv

def imshow(sample_element):
    plt.imshow(sample_element[0].numpy().reshape((128, 128)), cmap='gray')
    plt.show()

def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    batch_s = 10
    n_classes = 6

    all_transforms = transforms.Compose([transforms.ToTensor()])

    # Fingers dataset consists of 21600 128x128 black & white images of left/right hands
    train_dataset = Fingers('./data/train/*.png', transform=all_transforms)
    test_dataset = Fingers('./data/test/*.png', transform=all_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)
    
    classes = ('0', '1', '2', '3', '4', '5', '6')

if __name__ == "__main__":
    main()