import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
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

    all_transforms = transforms.Compose([transforms.ToTensor()])

    # Fingers dataset consists of 21600 128x128 black & white images of left/right hands
    train_dataset = Fingers('./data/train/*.png')
    test_dataset = Fingers('./data/test/*.png', transform=all_transforms)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    imshow(train_dataset[17])
    print(test_dataset[17])
   
    


if __name__ == "__main__":
    main()