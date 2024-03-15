#%% source used: https://github.com/havakv/pycox
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

# from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

import pandas as pd

#%% set seed
np.random.seed(1234)
_ = torch.manual_seed(1234)

#%% Load data

df_train = pd.read_excel('Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx', sheet_name='TCIA_n=197')


df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# sanity check
print(df_train.head())
# %% categorize columns

cols_standardize = ["age", "body_mass_index", "max_tumor_size", "NASH_score"]
cols_bin = ["sex", "major_comorbidity", "node_positive_primary", "synchronous_crlm", "multiple_metastases", "bilobar_disease", "extrahep_disease", "chemo_before_liver_resection", "preoperative_pve", "steatosis_yesno", "presence_sinusoidal_dilata", "NASH_yesno", "NASH_greater_4"]
cols_999 = ["clinrisk_score", "clinrisk_stratified", "carcinoembryonic_antigen", "fibrosis_greater_40_percent"]
cols_percent_999 = ["total_response_percent", "necrosis_percent", "fibrosis_percent", "mucin_percent"]
cols_scores_overall = ["overall_survival_months", "vital_status"]
cols_scores_rest = ["progression_or_recurrence", "months_to_DFS_progression", "vital_status_DFS", "progression_or_recurrence_liveronly", "months_to_liver_DFS_progression", "vital_status_liver_DFS"]

#%% For simplicity we will exclude all data that has 999 value, and only look at the overall survival months

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_bin]

x_mapper = DataFrameMapper(standardize + leave)

#%% Transform data and only use desired columns
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

#%% define labels

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

# get scores
durations_test, events_test = get_target(df_test)
#%% sanity check
print(labtrans.cuts)

#%% Define Neural Network
in_features = x_train.shape[1]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

#define network - simple structure is used to start with 
# architecture simply taken from the pycox example - idea was to change later
net = torch.nn.Sequential(
    torch.nn.Linear(in_features, 32),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(32),
    torch.nn.Dropout(0.1),
    
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(32),
    torch.nn.Dropout(0.1),
    
    torch.nn.Linear(32, out_features)
)

#%%
# define model, loss function is NLLLogistiHazard
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

batch_size = 64
epochs = 100
callbacks = [tt.cb.EarlyStopping()]

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)

#%%
_ = log.plot()

print(log.to_pandas().val_loss.min())

model.score_in_batches(val)

#%% interpolate to get smoother graphs
surv = model.interpolate(10).predict_surv_df(x_test)

surv.iloc[:, :].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
