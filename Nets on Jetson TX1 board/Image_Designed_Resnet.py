#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we tried to investigate simple ResNet  on Image data

# ### Importing the libraries

# In[1]:


import torch 

import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Function, Variable
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

#from pathlib import Path
import os
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
import time as time

import warnings


# #### Checking whether the GPU is active

# In[2]:


torch.backends.cudnn.enabled


# In[3]:


torch.cuda.is_available()


# #### Dataset paths

# In[4]:


test_path = "/home/saman/Saman/data/Image_Data01/test/Total/"
test_path


# ### Model parameters

# In[5]:


Num_Filter1= 16
Num_Filter2= 64
Ker_Sz1 = 5
Ker_Sz2 = 5

learning_rate= 0.0001

Dropout= 0.2
BchSz= 1
EPOCH= 5


# In[6]:


# Loss calculator
criterion = nn.CrossEntropyLoss()   # cross entropy loss


# In[7]:


transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (0.5,0.5,0.5)),    
])


# In[8]:


test_data = torchvision.datasets.ImageFolder(test_path,transform=transformation)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=BchSz, shuffle=True,
                                          num_workers=8)


# In[9]:


len(test_data)


# In[10]:


test_data.class_to_idx


# In[11]:


img = plt.imread(test_data.imgs[0][0]) # The second index if is 0 return the file name
IMSHAPE= img.shape
IMSHAPE


# ### Defining model

# ### Defining Resnet classes

# In[12]:


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# In[13]:


class ResNet(nn.Module):

    def __init__(self, block, layers, Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2, num_classes=2):
        self.in_channels = Num_Filter2
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels=Num_Filter1, kernel_size=Ker_Sz1, stride=1,
                               padding=int((Ker_Sz1-1)/2),bias=False)
        self.bn1 = nn.BatchNorm2d(Num_Filter1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.conv2 = nn.Conv2d(Num_Filter1, Num_Filter2, kernel_size=Ker_Sz2, stride=1,
                               padding=int((Ker_Sz2-1)/2),bias=False)
        self.bn2 = nn.BatchNorm2d(Num_Filter2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layer1 = self._make_layer(block, Num_Filter2, layers[0])
        self.layer2 = self._make_layer(block, Num_Filter2, layers[1], stride=1)
        
        self.maxpool = nn.MaxPool2d(7, stride=1, padding=1)
        self.fc = nn.Linear(12616704* block.expansion, num_classes)
        
        # Self initiation weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ### Finding number of parameter in our model

# In[14]:


def get_num_params(model):
    TotalParam=0
    for param in list(model.parameters()):
        nn=1
        for size in list(param.size()):
            nn = nn*size
        TotalParam += nn
    return TotalParam


# ### Testing function

# In[15]:


def test_model(model, test_loader):
    print("Starting testing...\n")
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    #with torch.no_grad():
    correct = 0
    total = 0
    test_loss_vect=[]
    test_acc_vect=[]

    since = time.time()  #record the beginning time

#     for i in range(10):

#         Indx = torch.randperm(len(test_data))
#         Cut=int(len(Indx)/10) # Here 10% showing the proportion of data is chosen for pooling
#         indices=Indx[:Cut]            
#         Sampler = Data.SubsetRandomSampler(indices)
#         pooled_data =  torch.utils.data.DataLoader(test_data , batch_size=BchSz,sampler=Sampler)

    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
              
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #predicted =outputs.data.max(1)[1]
        
        loss = criterion(outputs, labels)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    test_loss= loss / total
    test_accuracy= 100 * correct / total

    test_loss_vect.append(test_loss)
    test_acc_vect.append(test_accuracy)


#             print('Test accuracy and loss for the {}th pool: {:.2f} %, {:.5f}'
#                   .format(i+1, test_accuracy, test_loss))


    mean_test_loss = torch.mean(torch.stack(test_loss_vect))
#     mean_test_acc = np.mean(test_acc_vect)
#     std_test_acc = np.std(test_acc_vect)

#     print('-' * 10)
#     print('Average test accuracy on test data: %6.2f \nloss:%6.5f \nStandard deviion of accuracy:%6.4f'
#           %(mean_test_acc, mean_test_loss, std_test_acc))

    print('-' * 10)
    time_elapsed = time.time() - since
    print('Testing complete in %06.1f m %06.4f s' %(time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    print('Number of parameters in the model %d ' %(get_num_params(model)))

    return mean_test_loss


# ### Applying aumentation 

# In[16]:


model = ResNet(BasicBlock, [1, 1] , Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2) 
model = model.cuda()
#print(model)

# Defining optimizer with variable learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  


# In[17]:


get_num_params(model)


# In[18]:


testing = test_model (model, test_loader)


# In[ ]:




