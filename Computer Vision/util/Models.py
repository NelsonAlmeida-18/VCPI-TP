import torch
from torchinfo import summary
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import torchinfo


class Conv(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()

        self.maxpool1 = torch.nn.MaxPool2d(2)


        self.conv3 = torch.nn.Conv2d(32, 48, 3)
        self.bn3 = torch.nn.BatchNorm2d(48)
        self.relu3 = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(48, 48, 3)
        self.bn4 = torch.nn.BatchNorm2d(48)
        self.relu4 = torch.nn.ReLU()

        self.maxpool2 = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(1200, num_classes)
        

    def forward(self, x):    
        
        # input = (bs, 3, 32, 32)
        x = self.conv1(x) # -> (bs, 16, 26, 26)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x) # -> (bs, 32, 24, 24)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        
        x = self.conv3(x) # -> (bs, 48, 12, 12)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x) # -> (bs, 48, 10, 10)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        
        x = torch.flatten(x,1) # -> (bs, 48 * 5 * 5 = 1200)
        x = self.fc1(x)        # -> (bs, 10)

        return(x)



class Conv_II(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = torch.nn.ReLU()
        
        self.mp1 = torch.nn.MaxPool2d(2)
        self.dp1 = torch.nn.Dropout(0.25)
        
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.relu4 = torch.nn.ReLU()
        
        self.mp2 = torch.nn.MaxPool2d(2)
        self.dp2 = torch.nn.Dropout(0.25)
        
        # Calculate the flattened size after convolutions and pooling
        self.fc1 = torch.nn.Linear(256 * 8 * 8, 512)  # Assuming the input images are 32x32
        
        self.relu_fc1 = torch.nn.ReLU()
        self.d1 = torch.nn.Linear(512, 128)
        self.relu_d1 = torch.nn.ReLU()
        self.dp3 = torch.nn.Dropout(0.5)
        
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp1(x)
        x = self.dp1(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.mp2(x)
        x = self.dp2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.d1(x)
        x = self.relu_d1(x)
        x = self.dp3(x)
        x = self.fc2(x)
        
        return x
    

class ConvIII(torch.nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        #Lets do a simple Convolutional Neural Network with a kernel of 3x3
        self.conv1 = torch.nn.Conv2d(3, img_size, 3)
        self.bn1 = torch.nn.BatchNorm2d(img_size)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(img_size, img_size, 3)
        self.bn2 = torch.nn.BatchNorm2d(img_size)
        self.relu2 = torch.nn.ReLU()

        
        #With a 2x2 kernel
        self.maxpool1 = torch.nn.MaxPool2d(2) #Output size = img_size/2 ->16x16
        #Node dropouts
        self.dropout1 = torch.nn.Dropout(0.25)

        self.conv3 = torch.nn.Conv2d(img_size, img_size, 3)
        self.bn3 = torch.nn.BatchNorm2d(img_size)
        self.relu3 = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(img_size, img_size, 3)
        self.bn4 = torch.nn.BatchNorm2d(img_size)
        self.relu4 = torch.nn.ReLU()

        #max pooling
        self.maxpool2 = torch.nn.MaxPool2d(2) #output size = img_size/4 -> 8x8
        #dropout
        self.dropout2 = torch.nn.Dropout(0.25)
        #Another convolutional layer
        self.conv5 = torch.nn.Conv2d(img_size/4, img_size/4, 3)
        self.bn5 = torch.nn.BatchNorm2d(img_size/4)
        self.relu5 = torch.nn.ReLU()

        #
        self.maxpool3 = torch.nn.MaxPool2d(2) #output size = img_size/8 -> 4x4
        self.dropout3 = torch.nn.Dropout(0.25)

        #Fully connected layer
        self.fc1 = torch.nn.Linear(img_size*4*4, 512)
        self.relu_fc1 = torch.nn.ReLU()
        #Batch normalization
        self.bn6 = torch.nn.BatchNorm1d(512)
        #fully connected layer
        self.fc2 = torch.nn.Linear(512, num_classes)
        #self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        #input = (bs, 1, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.bn6(x)
        x = self.fc2(x)
        
        return x