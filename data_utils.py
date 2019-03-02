import tensorflow as tf
import pandas as pd


def scale(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) 

def process(df):
    df.fillna(0, inplace=True)
    df["buy_freq"] = df["buy_freq"].astype(int)   

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
        shuffle=shuffle)
    return inputs

