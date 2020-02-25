#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we tried to investigate simple  network on PDU data

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


test_path = "/home/saman/Saman/data/PDU_Raw_Data01/Test06_600x30/test/Total"
test_path


# ### Model parameters

# In[5]:


Num_Filter1= 16
Num_Filter2= 128
Ker_Sz1 = 5
Ker_Sz2 = 5

learning_rate= 0.0001

Dropout= 0.1
BchSz= 4
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

# ### Defining simple net model

# In[12]:


class ConvNet1(nn.Module):
    def __init__(self,  Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2, Dropout, num_classes=2):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(  
            nn.Conv2d(              # input shape (3, 30, 600)
                in_channels=3,      # input height
                out_channels=Num_Filter1,    # n_filters
                kernel_size=Ker_Sz1,      # filter size
                stride=1,           # filter movement/step
                padding=int((Ker_Sz1-1)/2), # if want same width and length of this image after con2d,
            ),                      # padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(Num_Filter1),     # Batch Normalization
            nn.ReLU(),              # Rectified linear activation
            nn.MaxPool2d(kernel_size=2, stride=2)) # choose max value in 2x2 area, 
                                                   # output shape (16, 75, 75)
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
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(Num_Filter2, 256, 
                      kernel_size=Ker_Sz2, 
                      stride=1, 
                      padding=int((Ker_Sz2-1)/2)),
            nn.BatchNorm2d(256),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape (64, 38, 38)
            nn.Dropout2d(p=Dropout))
        
        self.fc = nn.Linear(57600, num_classes) # fully connected layer, output 2 classes
        
    def forward(self, x):                  # Forwarding the data to classifier 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ### Finding number of parameter in our model

# In[13]:


def get_num_params(model):
    TotalParam=0
    for param in list(model.parameters()):
        nn=1
        for size in list(param.size()):
            nn = nn*size
        TotalParam += nn
    return TotalParam


# ### Testing function

# In[14]:


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


#             print('Test accuracy and loss for the {}th pool: {:.2f} %, {:.5f}'
#                   .format(i+1, test_accuracy, test_loss))


    mean_test_loss = torch.mean(torch.stack(test_loss_vect))

#     print('-' * 10)
#     print('Average test accuracy on test data: %6.2f \nloss:%6.5f \nStandard deviion of accuracy:%6.4f'
#           %(mean_test_acc, mean_test_loss, std_test_acc))

    print('-' * 10)
    time_elapsed = time.time() - since
    print('Testing complete in %06.1f m %06.4f s' %(time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    print('Number of parameters in the model %d ' %(get_num_params(model)))

    return mean_test_loss


# In[15]:


model = ConvNet1(Num_Filter1 , Num_Filter2, Ker_Sz1, Ker_Sz2, Dropout, num_classes=2)
model = model.cuda()
#print(model)

# Defining optimizer with variable learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  


# In[16]:


get_num_params(model)


# In[17]:


testing = test_model (model, test_loader)


# In[ ]:




