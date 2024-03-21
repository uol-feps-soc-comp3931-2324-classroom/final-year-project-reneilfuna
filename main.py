import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import Fingers
from model import ConvNet

def imshow(sample_element):
    plt.imshow(sample_element[0].numpy().reshape((128, 128)), cmap='gray')
    plt.show()

def main():
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    batch_s = 36
    n_classes = 6
    learning_rate = 0.005
    n_epochs = 3

    preprocess = transforms.Compose([transforms.ToTensor()])

    # Fingers dataset consists of 21600 128x128 black & white images of left/right hands
    # Each image then providing a 1x128x128 dimensional tensor

    # Training set consists of 18,000 images
    train_dataset = Fingers('./data/train/*.png', transform=preprocess)
    # Test set consists of 2,600 images
    test_dataset = Fingers('./data/test/*.png', transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)
    
    classes = ('0', '1', '2', '3', '4', '5')

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader)
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimise
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')       

    with torch.no_grad(): # Don't need backward propagation and gradient calculations
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_s):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        accuracy = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {accuracy} %')
        
        for i in range(n_classes):
            if n_class_samples[i] != 0:
                accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
            else:
                accuracy = 0

            print(f'Accuracy of {classes[i]} fingers: {accuracy} %')


if __name__ == "__main__":
    main()