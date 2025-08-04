import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from constant import *
from func import *

data_csv = pd.read_csv(CSV_PATH, usecols=["ReceivedPower[dBm]"]) # csvを読み込みデータフレームに
true_data = data_csv.values.astype(np.float64)
normalized_true_data = (true_data - true_data.mean()) / true_data.std()

x,y=make_data_set(normalized_true_data,INPUT_LEN)

if os.path.exists(MODEL_PATH):
    print("✅ 既存のモデルを読み込みます")
    model = load_model(MODEL_PATH)
else:
    print("モデルが見つかりません")

predicted=model.predict(x)
predicted =  predicted * true_data.std() + true_data.mean()
'''
##### ここから再帰予測 #####
input_data=normalized_true_data[-PREDICT_LEN-INPUT_LEN:-PREDICT_LEN]
future_result = np.empty((0,)) # (0,)で空の配列になる

for i in range(PREDICT_LEN):
    print(f'step:{i+1}')
    test_data=np.reshape(input_data,(1,INPUT_LEN,1))
    predicted=model.predict(test_data)
    predicted =  predicted * true_data.std() + true_data.mean()

    input_data=np.delete(input_data,0)
    input_data=np.append(input_data,predicted)

    future_result=np.append(future_resul,predicted)
mse=np.mean((future_result - true_data[-PREDICT_LEN:])**2)
rmse=np.sqrt(np.mean((future_result - true_data[-PREDICT_LEN:])**2))
print(f"2乗誤差:{rmse}")
'''

plt.figure()
plt.plot(range(INPUT_LEN,PLOT_RANGE+INPUT_LEN), predicted[:PLOT_RANGE], color="g", label="future_predict")
plt.plot(range(0,PLOT_RANGE+INPUT_LEN),true_data[:PLOT_RANGE+INPUT_LEN],color="r",alpha=0.5,label="true_data")
#plt.plot(range(len(true_data)-PREDICT_LEN, len(true_data)), future_result, color="g", label="future_predict")
#plt.plot(range(len(true_data)-PLOT_RANGE-PREDICT_LEN,len(true_data)),true_data[-PLOT_RANGE-PREDICT_LEN:],color="r",alpha=0.5,label="true_data")
plt.legend()
plt.show()
