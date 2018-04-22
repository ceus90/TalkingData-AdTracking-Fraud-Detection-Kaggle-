import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# style.use('fivethirtyeight')
df_train_sample = pd.read_csv("~/Downloads/train_sample.csv")
df_train = pd.read_csv("~/Downloads/train.csv", nrows=10000000)

a = df_train.head(2).values
print a.shape
print df_train.isnull().sum()
print df_train.dtypes

df_train['click_time'] = pd.to_datetime(df_train['click_time'])
df_train['attributed_time'] = pd.to_datetime(df_train['attributed_time'])

df_train.groupby('is_attributed').agg({'attributed_time':'unique'})

df_plot1 = df_train.groupby(df_train['is_attributed'])['is_attributed'].count()
df_plot1.plot.bar(fontsize=8, legend=True)

df_train['days_to_download'] = df_train['attributed_time'] - df_train['click_time']
days_to_download = df_train[~df_train['days_to_download'].isnull()]['days_to_download']
hours_to_download = days_to_download.dt.components.hours
hours_to_download[hours_to_download != 0].plot(kind='hist' ,bins=10)

print df_train[df_train['is_attributed']==1].head()