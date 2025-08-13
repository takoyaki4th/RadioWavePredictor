import numpy as np
import pandas as pd
import os
path = os.path.dirname(__file__)
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array
from setting import INPUT_LEN,BATCH_SIZE

def normalize(data):
    return (data - data.mean()) / data.std()
    
def denormalize(normalized_data,base_data):
    return normalized_data * base_data.std() + base_data.mean()

#### ここから↓クソコード注意 ごめんなさい ####

def read_csv_and_convert_dataset(csv_path): 
    csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"]) # csvを読み込みデータフレームに
    data_numpy = csv_data.values.astype(np.float64)
    data_numpy_normalized = normalize(data_numpy)
    
    #datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている    
    dataset=timeseries_dataset_from_array(
        data_numpy_normalized,
        targets=data_numpy_normalized[INPUT_LEN:],
        sequence_length=INPUT_LEN,
        batch_size=None,
        shuffle=None
    )
    return dataset 

#csvからデータを読み込んで機械学習の訓練データセットと検証データセットを返す関数
def load_training_data(training_cources,validation_cource,learn_mode):
    #各コースの訓練データ群の配列
    train_dataset_arr=[]
    for cource in training_cources:
        csv_path = f"{path}/result/WAVE{cource:04d}/result_n{learn_mode}-001.csv" 
        train_dataset_i=read_csv_and_convert_dataset(csv_path)

        train_dataset_arr.append(train_dataset_i)

    # train_dataset_arrの中身をすべてつなげる
    train_dataset = train_dataset_arr[0]
    for ds in train_dataset_arr[1:]:
        train_dataset = train_dataset.concatenate(ds)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    csv_path = f"{path}/result/WAVE{validation_cource:04d}/result_nt-001.csv" 
    val_dataset=read_csv_and_convert_dataset(csv_path)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset,val_dataset

#dataからRNN用のデータセットを生成する関数
#timeseries_dataset_from_array()と使い分ける
#make_data_setはnumpy配列を返すので、後から加工しやすい
#timeseries_dataset_from_arrayはtf.data.Datasetオブジェクトを返すので、prefetchなどtensorflow専用の関数が使える
def make_data_set(changed_data,input_len):
    data,target=[],[]
    
    for i in range(len(changed_data)-input_len):
        data.append(changed_data[i:i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)
