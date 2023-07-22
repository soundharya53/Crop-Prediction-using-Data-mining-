import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable
import matplotlib.pyplot as plt
path =r"C:\Users\sound\source\repos\cropprediction\cropprediction\FinalDataset.csv"
training_set = pd.read_csv(path)
training_set = np.array(training_set)
idnum = int(max(training_set[:,0]))
param = int(max(training_set[:,1]))
def convert(data):
    new_data = []
    for paramnum in range(1, idnum+1):
        paramid = data[:,1][data[:,0] == paramnum]
        rate = data[:,2][data[:,0] == paramnum]
        ratings = np.zeros(param)
        ratings[paramid - 1] = rate
        new_data.append(list(ratings))
    return np.array(new_data)
training_set = torch.FloatTensor(convert(training_set))
def convert_likes(df):
    df[df == 0] = -1
    df[df == 1] = 0
    df[df == 2] = 0
    df[df >=3 ] = 1
    return df
training_set = convert_likes(training_set)
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t())
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk, lr=0.01):
        self.W += lr * torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += lr * torch.sum((v0 - vk), 0)
        self.a += lr * torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 16
batch_size = 32
rbm = RBM(nv, nh)
nb_epoch = 10
lr = 0.03
losses = []

for epoch in range(1, nb_epoch + 1):
    
    train_loss = 0
    epoch_loss = []
    s = 0.0
    
    for idnum in range(0, idnum - batch_size, batch_size):
        
        vk = training_set[idnum:idnum+batch_size]
        v0 = training_set[idnum:idnum+batch_size]
        
        ph0,_ = rbm.sample_h(v0)
        for k in range(100):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
            
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk, lr)
        
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        epoch_loss.append(train_loss/s)
        s += 5.0
        
    losses.append(epoch_loss[-1])
    if(epoch %2 == 0):
        print('Epoch:{0:4d} Train Loss:{1:1.4f}'.format(epoch, train_loss/s))
