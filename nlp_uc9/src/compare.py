#!/usr/bin/env python3
import pandas as pd
from pycaret.classification import *

import warnings
warnings.filterwarnings('ignore')


experiments = {"Metrics Only" : "../data/ml_ready/train_metrics_only.csv", 
               "Text Only" : "../data/ml_ready/train_texts_only.csv",
               "Combined" : "../data/ml_ready/train_combined.csv"
#               "Filtered min_df=0.05 max_df=0.70" : "../data/prepared/train_vectorized_filtered.csv"
}

df_list = []

for experiment, data in experiments.items():
    print(experiment, data)

    df_train = pd.read_csv(data)  
    
    s = setup(data=df_train, target="RECALL",session_id=365, fold=5, silent=True, verbose=False)
    best = compare_models()
    predict_model(best)

    df_results = pull().reset_index(drop=True)
    df_results["Experiment"] = experiment
    df_list.append(df_results)
    

df = pd.concat(df_list, axis=0, ignore_index=True)
df.to_csv("../results/compare.csv", index=False)

print(df) 
    
