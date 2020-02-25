#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we tried to investigate simple Inception network on image data

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


# ### Defining models

# #### Defining a class of our simple model

# In[12]:


class ConvNet(nn.Module):
    def __init__(self,  Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2,  Dropout, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(  
            nn.Conv2d(              # input shape (3, 30, 600)
                in_channels=3,      # input height
                out_channels=Num_Filter1,    # n_filters
                kernel_size=Ker_Sz1,      # Kernel size
                stride=1,           # filter movement/step
                padding=int((Ker_Sz1-1)/2), # if want same width and length of this image after con2d,
            ),                              # padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(Num_Filter1),     # Batch Normalization
            nn.ReLU(),              # Rectified linear activation
            nn.MaxPool2d(kernel_size=2, stride=2)) # choose max value in 2x2 area, 
                                                   
        # Visualizing this in https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(Num_Filter1, Num_Filter2, 
                      kernel_size=Ker_Sz2, 
                      stride=1, 
                      padding=int((Ker_Sz2-1)/2)),
            nn.BatchNorm2d(Num_Filter2),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape (64, 38, 38)
            nn.Dropout2d(p=Dropout))
        
        self.fc = nn.Linear(1050*Num_Filter2, num_classes) # fully connected layer, output 2 classes

        
        
    def forward(self, x):                  # Forwarding the data to classifier 
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # flatten the output of conv2 to (batch_size, 64*38*38)
        out = self.fc(out)
        return out


# ### Defining inception classes

# In[13]:


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return x


# In[14]:


class Inception(nn.Module):

    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


# In[15]:


class Inception_Net(nn.Module):
    def __init__(self,  Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2,  Dropout, num_classes=2):
        super(Inception_Net, self).__init__()
        self.layer1 = nn.Sequential(  
            nn.Conv2d(              # input shape (3, 30, 600)
                in_channels=3,      # input height
                out_channels=Num_Filter1,    # n_filters
                kernel_size=Ker_Sz1,      # Kernel size
                stride=1,           # filter movement/step
                padding=int((Ker_Sz1-1)/2), # if want same width and length of this image after con2d,
            ),                              # padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(Num_Filter1),     # Batch Normalization
            nn.ReLU(),              # Rectified linear activation
            nn.MaxPool2d(kernel_size=2, stride=2)) # choose max value in 2x2 area, 
                                                   
        # Visualizing this in https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(Num_Filter1, Num_Filter2, 
                      kernel_size=Ker_Sz2, 
                      stride=1, 
                      padding=int((Ker_Sz2-1)/2)),
            nn.BatchNorm2d(Num_Filter2),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape (64, 38, 38)
            nn.Dropout2d(p=Dropout))
        
        self.Inception = Inception(Num_Filter2)
            
        self.fc = nn.Linear(1705984, num_classes) # fully connected layer, output 2 classes
        
    def forward(self, x):                  # Forwarding the data to classifier 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.Inception(out)
        out = out.view(out.size(0), -1) # flatten the output of conv2 to (batch_size, 64*38*38)
        out = self.fc(out)
        return out


# ### Finding number of parameter in our model

# In[16]:


def get_num_params(model):
    TotalParam=0
    for param in list(model.parameters()):
        nn=1
        for size in list(param.size()):
            nn = nn*size
        TotalParam += nn
    return TotalParam


# ### Testing function

# In[17]:


def test_model(model, test_loader):
    print("Starting testing...\n")
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    #with torch.no_grad():
    correct = 0
    total = 0
    test_loss_vect=[]
    test_acc_vect=[]

    since = time.time()  #record the beginning time


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

    return  mean_test_loss


# ### Applying aumentation 

# In[18]:


transformation2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (0.5,0.5,0.5)),    
]) 


# In[19]:


test_data = torchvision.datasets.ImageFolder(test_path,transform=transformation2)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=BchSz, shuffle=True,
                                          num_workers=8)


# In[20]:


model = Inception_Net(Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2, Dropout, num_classes=2)

model = model.cuda()
#print(model)

# Defining optimizer with variable learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  


# In[21]:


get_num_params(model)


# In[22]:


testing = test_model (model, test_loader)


# In[ ]:




