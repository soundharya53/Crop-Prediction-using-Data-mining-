import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable
import matplotlib.pyplot as plt


df = pd.read_csv("FinalDataset.csv")
df1=df.drop_duplicates()
pr=df1['Production'].mode()[0]
t=df1['Temperature'].mean()
h=df1['humidity'].mean()
ph=df1['ph'].mean()
r=df1['rainfall'].mean()
df1['Production']=df1['Production'].fillna(value=pr)
df1['Temperature']=df1['Temperature'].fillna(value=t)
df1['humidity']=df1['humidity'].fillna(value=h)
df1['ph']=df1['ph'].fillna(value=ph)
df1['rainfall']=df1['rainfall'].fillna(value=r)
df1=df1.dropna()  
df1.to_csv('modified_dataset.csv')
print("Mode of Production:",pr)
print("Mean of Temperature:",t)
print("Mean of Humidity:",h)
print("Mean of pH:",ph)
print("Mean of Rainfall:",r)
