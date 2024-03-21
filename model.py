import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            # Convolutional layers
            # Images are all in grayscale so only 1 colour channel
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1)
            self.pool = nn.MaxPool2d(2,2)
            # Fully Connected Layer
            self.fc1 = nn.Linear(128*13*13, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 6)


        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            # Flattening the tensor dimensions after convolutions
            x = x.view(-1, 128*13*13)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            

            