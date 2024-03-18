import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import Fingers

def imshow(sample_element):
    plt.imshow(sample_element[0], cmap='gray')
    plt.show()

def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = './data/test/*.png'
dataset = Fingers(directory=train_dir)
imshow(dataset[128])

if __name__ == "__main__":
    main()