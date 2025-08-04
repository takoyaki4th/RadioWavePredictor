import numpy as np
import pandas as pd
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
'''
def output_txt(arr,path):
    # テキストファイルに書き込み
    with open(path, "w") as f:
        for row in arr:
            f.write(" ".join(map(str, row)) + "\n")
'''