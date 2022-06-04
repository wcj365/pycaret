#!/usr/bin/env python3
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


df_y = pd.read_csv("../data/source/y_train.csv")
df_y.rename(columns={"recall":"RECALL"}, inplace=True)

# Read the first row to get the column headers
columns = pd.read_csv("../data/source/X_train.csv", nrows=1)

# Read all columns except for the MDR_text column
df_metrics = pd.read_csv("../data/source/X_train.csv", usecols=[x for x in columns if x != "MDR_text"])
df_metrics_only = pd.concat([df_y, df_metrics], axis=1)
df_metrics_only.to_csv("../data/ml_ready/train_metrics_only.csv", index=False)

df_texts = pd.read_csv("../data/prepared/train_chi_squared_p50.csv")
df_texts_only = pd.concat([df_y, df_texts], axis=1)
df_texts_only.to_csv("../data/ml_ready/train_texts_only.csv", index=False)

df_combined = pd.concat([df_y, df_metrics, df_texts], axis=1)
df_combined.to_csv("../data/ml_ready/train_combined.csv", index=False)