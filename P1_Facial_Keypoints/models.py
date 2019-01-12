import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding = 2)
        I.xavier_normal_(self.conv1.weight, gain=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, padding = 2)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, padding = 2)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, padding = 1)
        self.conv5 = nn.Conv2d(96, 128, kernel_size=3, padding = 1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(128*14*14, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.dropout25 = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)

        
    def forward(self, x):
        # Size changes from (1, 224, 224) to (32, 224, 224)
        x = F.relu(self.conv1(x))

        # Size changes from (32, 224, 224) to (32, 112, 112)
        x = self.pool(x)

        # Size changes from (32, 112, 112) to (48, 56, 56)
        x = self.pool(F.relu(self.conv2(x)))

        # Size changes from (48, 56, 56) to (64, 28, 28)
        x = self.pool(F.relu(self.conv3(x)))

        # Size changes from (64, 28, 28) to (96, 14, 14)    
        x = self.pool(F.relu(self.conv4(x)))

        # Size changes from (96, 14, 14) to (128, 14, 14)  
        x = F.relu(self.conv5(x))

        # Size changes from (128, 14, 14) to (25088)
        x = x.view(x.size(0), -1)
        
        # Size changes from (128*14*14) to (1024)
        x = F.relu(self.fc1(x))
        x = self.dropout25(x)
        
        # Size changes from (1024) to (512)
        x = F.relu(self.fc2(x))
        x = self.dropout25(x)
        
        # Size changes from (512) to (136)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
