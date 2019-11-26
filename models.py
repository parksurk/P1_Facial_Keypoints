## TODO: define the convolutional neural network architecture

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

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # NaimishNet implementation (https://arxiv.org/pdf/1710.00977.pdf "Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet")

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)

        # Max-Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        # FC layers
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=136)

        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)

        # Custom weights initialization
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        # Conv layers have weights initialized with random # drawn from uniform distribution
        #        m.weight = nn.init.uniform(m.weight, a=0, b=1)
        #    elif isinstance(m, nn.Linear):
        #        # FC layers have weights initialized with Glorot uniform initialization
        #        m.weight = nn.init.xavier_uniform(m.weight, gain=1)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))


        ## Conv layers - Feature Extracting
        ## Output Feature maps Image Size Formula:(W − F + 2P ) / S + 1
        x = self.conv1(x)       # (224 - 4 + 2*0) / 1 + 1 = 221
        x = self.pool1(x)       # (221 - 2 + 2*0) / 2 + 1 = 109.5 + 1 = 110
        x = F.elu(x)
        x = self.dropout1(x)

        x = self.conv2(x)       # (110 - 3 + 2*0) / 1 + 1 = 108
        x = self.pool2(x)       # (108 - 2 + 2*0) / 2 + 1 = 53 + 1 = 54
        x = F.elu(x)
        x = self.dropout2(x)

        x = self.conv3(x)       # (54 - 4 + 2*0) / 1 + 1 = 53
        x = self.pool3(x)       # (53 - 2 + 2*0) / 2 + 1 = 25.5 + 1 = 26
        x = F.elu(x)
        x = self.dropout3(x)

        x = self.conv4(x)       # (26 - 4 + 2*0) / 1 + 1 = 26
        x = self.pool4(x)       # (26 - 2 + 2*0) / 2 + 1 = 12 + 1 = 13
        x = F.elu(x)
        x = self.dropout4(x)

        ## Flatten - Connction with Conv layers and FC layers
        x = x.view(x.size(0), -1)   # 13 * 13 * 256 = 43264

        ## Fully connected layers
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = F.tanh(x) # different activation function with tanh
        x = self.dropout6(x)

        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

class NetWithBatchNormalization(nn.Module):

    def __init__(self):
        super(NetWithBatchNormalization, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # NaimishNet implementation (https://arxiv.org/pdf/1710.00977.pdf "Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet")

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)

        # Max-Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        # FC layers
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=136)

        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)

        # Custom weights initialization
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        # Conv layers have weights initialized with random # drawn from uniform distribution
        #        m.weight = nn.init.uniform(m.weight, a=0, b=1)
        #    elif isinstance(m, nn.Linear):
        #        # FC layers have weights initialized with Glorot uniform initialization
        #        m.weight = nn.init.xavier_uniform(m.weight, gain=1)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))


        ## Conv layers - Feature Extracting
        ## Output Feature maps Image Size Formula:(W − F + 2P ) / S + 1
        x = self.conv1(x)       # (224 - 4 + 2*0) / 1 + 1 = 221
        x = self.pool1(x)       # (221 - 2 + 2*0) / 2 + 1 = 109.5 + 1 = 110
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout1(x)

        x = self.conv2(x)       # (110 - 3 + 2*0) / 1 + 1 = 108
        x = self.pool2(x)       # (108 - 2 + 2*0) / 2 + 1 = 53 + 1 = 54
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout2(x)

        x = self.conv3(x)       # (54 - 4 + 2*0) / 1 + 1 = 53
        x = self.pool3(x)       # (53 - 2 + 2*0) / 2 + 1 = 25.5 + 1 = 26
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout3(x)

        x = self.conv4(x)       # (26 - 4 + 2*0) / 1 + 1 = 26
        x = self.pool4(x)       # (26 - 2 + 2*0) / 2 + 1 = 12 + 1 = 13
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout4(x)

        ## Flatten - Connction with Conv layers and FC layers
        x = x.view(x.size(0), -1)   # 13 * 13 * 256 = 43264

        ## Fully connected layers
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = F.tanh(x) # different activation function with tanh
        x = self.dropout6(x)

        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
