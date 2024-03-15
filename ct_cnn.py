#%% source used: https://github.com/havakv/pycox
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F

# from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

import pandas as pd

#%%
np.random.seed(1234)
_ = torch.manual_seed(1234)

#%% Load the full dataset from the list file

# with open("list_data.txt", "rb") as file:
#     # Use the pickle.load() function to load the list from the file
#     subject_data_list_test = pickle.load(file)


# # Fill in missing data with zeros

# im_height = 512 # len(subject_data_list_test[0][0])
# im_width = 512 # len(subject_data_list_test[0][0][0])

# l = []
# for i in range(len(subject_data_list_test)):
#     l.append(len(subject_data_list_test[i]))

# max_sequence = max(l)

# blank_im_list = [[0]*im_width for _ in range(im_height)]

# for i in range(len(subject_data_list_test)):
#     if len(subject_data_list_test[i]) < max_sequence:
#         for j in range(max_sequence - len(subject_data_list_test[i])):
#             subject_data_list_test[i].append(blank_im_list)

# #%%

# # Split subject_data_list_test into train, test, and validation sets
# test_size = int(np.round(0.2 * len(subject_data_list_test)))
# val_size =  int(np.round(0.2 * 0.8 * len(subject_data_list_test)))
# train_size = len(subject_data_list_test) - val_size - test_size

# x_train = subject_data_list_test[:train_size]
# x_test = subject_data_list_test[train_size:train_size+test_size]
# x_val = subject_data_list_test[train_size+test_size:]

#%% Load array from file
subject_data = np.load('image_data.npy')

#%% 

# Split subject_data into train, test, and validation sets
test_size = int(np.round(0.2 * subject_data.shape[0]))
val_size =  int(np.round(0.2 * 0.8 * subject_data.shape[0]))
train_size = subject_data.shape[0] - val_size - test_size

x_train = subject_data[:train_size]
x_test = subject_data[train_size:train_size+test_size]
x_val = subject_data[train_size+test_size:]

#%% Get lables: 

df_train = pd.read_excel('Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx', sheet_name='TCIA_n=197')

df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

num_durations = 10

# define lable transform
labtrans = LogisticHazard.label_transform(num_durations)

# get scores
get_target = lambda df: (df['overall_survival_months'].values, df['vital_status'].values)

# discretize the scores
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

durations_test, events_test = get_target(df_test)

#%% Define CNN
out_features = num_durations

# architecture simply taken from the pycox example - idea was to change later
class Net(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)  
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1, 16) 
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.glob_avg_pool(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(labtrans.out_features)

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
#%% Use dataloader 

# LogisticHazard needs to be used since it can handle the censored data, the format of the lables in use, and it has 
# an apropriate loss function. However the it need the input data to be on a specific format, I tried to handle this quickly
# by simply using the same approach as in the mentioned example, 

x_train_tensor = torch.from_numpy(x_train).float()
x_val_tensor =   torch.from_numpy(x_val).float()

class SimDatasetSingle(torch.utils.data.Dataset):
    def __init__(self, dataset, time, event):
        self.dataset = dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
        img = self.dataset[index][0]
        return img, (self.time[index], self.event[index])
    
dataset_train = SimDatasetSingle(x_train_tensor, *y_train)
dataset_val = SimDatasetSingle(x_val_tensor, *y_val)

def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()

dl_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, collate_fn=collate_fn)
dl_val =  torch.utils.data.DataLoader(dataset_val, shuffle=False, collate_fn=collate_fn)

#%% Train model
callbacks = [tt.cb.EarlyStopping(patience=5)]
epochs = 50
verbose = True

# Your code here
log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_val)
_ = log.plot()


#%% get survival prob. - interpolate to get smoother graphs

# Survival curves don't look correct, this code was just copy pasted and should probably be modified
class SimInput(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        return img

x_test_tensor =   torch.from_numpy(x_test).float()
dataset_test = SimInput(x_test_tensor)

dl_test_x =  torch.utils.data.DataLoader(dataset_test, shuffle=False)

surv = model.predict_surv_df(dl_test_x)


# survival curves do not look correct
# this is clear since the dimensions of surv are wrong,
# this might be because of batch size or something else
surv.iloc[:, :41].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#%%