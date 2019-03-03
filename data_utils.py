import tensorflow as tf
import pandas as pd
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
import numpy as np


def scale(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) 


def read_data(data_file, num_epochs, shuffle):
    df_data = pd.read_csv(
    tf.gfile.Open(data_file),
    header=0,
    skipinitialspace=True,
    engine="python")
    process(df_data)
    if "test" in data_file:
        labels = None
    else:    
        labels = df_data["y_buy"]
    inputs = tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=50000,
        num_threads=1)
    return inputs

# 采样
def read_data_with_sampling(data_file, num_epochs, shuffle):
    df_data = pd.read_csv(
    tf.gfile.Open(data_file),
    header=0,
    skipinitialspace=True,
    engine="python")
    process(df_data)
    X = df_data.drop(["y_buy"], axis = 1)
    y = df_data["y_buy"]
    X_column = X.columns
    y_column = ["y_buy"]
    print(sorted(Counter(y).items()))
    # 直接欠采样
    # nm = NearMiss(random_state=0, version=3)
    # X_resampled, y_resampled = nm.fit_sample(X, y)
    # print(sorted(Counter(y_resampled).items()))
    # 欠采样和过采样结合1
    smote_enn = SMOTEENN(sampling_strategy=0.5, random_state=0)
    X_resampled, y_resampled = smote_enn.fit_sample(X, y)
    print(sorted(Counter(y_resampled).items()))
    ## 欠采样和过采样结合2
    ##smote_tomek = SMOTETomek(sampling_strategy=0.1, random_state=0) 
    ##X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    ##print(sorted(Counter(y_resampled).items()))
    X_resampled = pd.DataFrame(X_resampled, columns=X_column) 
    y_resampled = pd.DataFrame(y_resampled, columns=y_column) 
    process(X_resampled)
    inputs = tf.estimator.inputs.pandas_input_fn(
        X_resampled,
        y_resampled,
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=70000,
        num_threads=1)
    return inputs

def process(df):
    df.fillna(0, inplace=True)
    df["buy_freq"] = df["buy_freq"].astype(int)   
    df["visit_freq"] = df["visit_freq"].astype(int)   
    df["last_buy"] = df["last_buy"].astype(int)   
    df["last_visit"] = df["last_visit"].astype(int)   
    df["multiple_buy"] = df["multiple_buy"].astype(int)   
    df["multiple_visit"] = df["multiple_visit"].astype(int)   
    df["uniq_urls"] = df["uniq_urls"].astype(int)   
    df["num_checkins"] = df["num_checkins"].astype(int)   
