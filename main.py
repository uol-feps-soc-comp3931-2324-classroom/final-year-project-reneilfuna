import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import Fingers

def imshow(sample_element):
    plt.imshow(sample_element[0], cmap='gray')
    plt.show()

def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    batch_size = 64
    n_classes = 6

    # Fingers dataset consists of 21600 128x128 black & white images of left/right hands
    train_dataset = Fingers('./data/train/*.png')
    test_dataset = Fingers('./data/test/*.png')

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    


if __name__ == "__main__":
    main()