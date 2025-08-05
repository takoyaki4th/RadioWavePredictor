import numpy as np
import pandas as pd
import os
path = os.path.dirname(__file__)
from constant import INPUT_LEN

def make_data_set(changed_data,input_len):
    data,target=[],[]
    
    for i in range(len(changed_data)-input_len):
        data.append(changed_data[i:i + input_len])
        target.append(changed_data[i + input_len])

    # LSTM用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

def read_csv_and_convert_dataset(csv_path): 
    csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"]) # csvを読み込みデータフレームに
    data_numpy = csv_data.values.astype(np.float64)
    data_numpy_normalized = (data_numpy - data_numpy.mean()) / data_numpy.std()
    x,y=make_data_set(data_numpy_normalized,INPUT_LEN)
    return x,y

def load_training_data(training_cources,validation_cource,learn_mode):
    #各コースの訓練データ群の配列
    train_x_arr=[]
    train_y_arr=[]

    for cource in training_cources:
        csv_path = f"{path}/result/WAVE{cource:04d}/result_n{learn_mode}-001.csv" 
        train_x_i,train_y_i=read_csv_and_convert_dataset(csv_path)

        train_x_arr.append(train_x_i)
        train_y_arr.append(train_y_i)

    csv_path = f"{path}/result/WAVE{validation_cource:04d}/result_nt-001.csv" 
    val_x,val_y=read_csv_and_convert_dataset(csv_path)

    return (train_x_arr,train_y_arr),(val_x,val_y)
