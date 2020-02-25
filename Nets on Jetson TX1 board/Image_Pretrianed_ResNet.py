#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we changed the network to pretrianed VGG

# ### Importing the libraries

# In[1]:


import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Function, Variable
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


#from pathlib import Path
import os
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
import time as time


# #### Checking whether the GPU is active

# In[2]:


torch.backends.cudnn.enabled


# In[3]:


torch.cuda.is_available()


# ### Hyper Parameters

# In[4]:


EPOCH= 5

BchSz=1 # BATCHSIZE

# Learning Rate
learning_rate = 0.0001


# Dropout rate
Dropout=0.1


# In[5]:


# Loss calculator
criterion = nn.CrossEntropyLoss()   # cross entropy loss


# Dataset paths

# In[6]:


test_path = "/home/saman/Saman/data/Image_Data01/test/Total/"
test_path 


# ### Loading data set (including augmentations)

# #### Train the model with vertically and horizontally flipping and normalization

# In[7]:


transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (0.5,0.5,0.5)),    
]) 


# ### Reading data after transformation

# In[8]:


test_data = torchvision.datasets.ImageFolder(test_path,transform=transformation)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=BchSz, shuffle=True,
                                          num_workers=8)


# ### Checking dataset

# In[9]:


len(test_data)


# In[10]:


test_data.class_to_idx


# In[11]:


img = plt.imread(test_data.imgs[0][0]) # The second index if is 0 return the file name
IMSHAPE= img.shape
IMSHAPE


# ### Training and Validating

# ### Testing function

# In[12]:


def get_num_params(model):
    TotalParam=0
    for param in list(model.parameters()):
        nn=1
        for size in list(param.size()):
            nn = nn*size
        TotalParam += nn
    return TotalParam


# In[13]:


def test_model(model, test_loader):
    print("Starting testing...\n")
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    #with torch.no_grad():
    correct = 0
    total = 0
    test_loss_vect=[]
    test_acc_vect=[]

    since = time.time()  #record the beginning time
        
#         for i in range(10):
            
#             Indx = torch.randperm(len(test_data))
#             Cut=int(len(Indx)/10) # Here 10% showing the proportion of data is chosen for pooling
#             indices=Indx[:Cut]            
#             Sampler = Data.SubsetRandomSampler(indices)
#             pooled_data =  torch.utils.data.DataLoader(test_data , batch_size=BchSz,sampler=Sampler)

    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        total += labels.size(0)
        #correct += (predicted == labels).sum().item()

    test_loss= loss / total
    #test_accuracy= 100 * correct / total

    test_loss_vect.append(test_loss)
    #test_acc_vect.append(test_accuracy)


#             print('Test accuracy and loss for the {}th pool: {:.2f} %, {:.5f}'
#                   .format(i+1, test_accuracy, test_loss))


    mean_test_loss = torch.mean(torch.stack(test_loss_vect))    #mean_test_acc = np.mean(test_acc_vect)
    #std_test_acc = np.std(test_acc_vect)

    #         print('-' * 10)
    #         print('Average test accuracy on test data: {:.2f} %, loss: {:.5f}, Standard deviion of accuracy: {:.4f}'
    #               .format(mean_test_acc, mean_test_loss, std_test_acc))
    
   
    print('-' * 10)
    time_elapsed = time.time() - since
    print('Testing complete in {:.1f}m {:.4f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('-' * 10)
    print('-' * 10)
    print('Number of parameters in the model %d ' %(get_num_params(model)))

    return mean_test_loss # , mean_test_loss, std_test_acc


# ### Resnet model 

# #### Pretrained Resnet  model

# In[14]:


model = models.resnet101(pretrained=False) #2 is number of classes

fc = nn.Linear(in_features=2048, out_features=2)
model.fc= fc

#print(model)


# In[15]:


model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# In[16]:


get_num_params(model)


# In[17]:


testing = test_model (model, test_loader)


# In[ ]:




