import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from constant import *
from func import *

data_csv = pd.read_csv(CSV_PATH, usecols=["ReceivedPower[dBm]"]) # csvを読み込みデータフレームに
true_data = data_csv.values.astype(np.float64)
normalized_true_data = (true_data - true_data.mean()) / true_data.std()

input_len=[5,25,50,100]
x=[]
y=[]
for len in input_len:
    x_i,y_i=make_data_set(normalized_true_data,len)
    x.append(x_i)
    y.append(y_i)

predicteds=[]
for i in range(0,4):
    model_path=path+f"/cource_{COURCE_NUM}_model_{input_len[i]}.h5"
    if os.path.exists(model_path):
        print("✅ 既存のモデルを読み込みます")
        model = load_model(model_path)
    else:
        print("モデルが見つかりません")

    predicted=model.predict(x[i])
    predicted =  predicted * true_data.std() + true_data.mean()
    predicteds.append(predicted)

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

for i in range(0,4):
    rmse=np.sqrt(np.mean((predicteds[i]-true_data[input_len[i]:])**2))
    print(i+1,rmse)

x=[]
x.append(np.linspace(0,20,400))
x.append(np.linspace(5/20,20,395))
x.append(np.linspace(25/20,20,375))
x.append(np.linspace(50/20,20,350))
x.append(np.linspace(100//20,20,300))


plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("ReceivedPower[dBm]")
#plt.plot(range(INPUT_LEN,PLOT_RANGE+INPUT_LEN), predicted[:PLOT_RANGE], color="g", label="future_predict")
plt.plot(x[0],true_data[:400],color="r",alpha=0.5,label="true_data")
plt.plot(x[1], predicteds[0][:395], color="y", alpha=0.5,label="5")
plt.plot(x[2], predicteds[1][:375], color="b", alpha=0.5,label="25")
plt.plot(x[3], predicteds[2][:350], color="c", alpha=0.5,label="50")
plt.plot(x[4], predicteds[3][:300], color="g", alpha=0.5,label="100")

#plt.plot(range(len(true_data)-PREDICT_LEN, len(true_data)), future_result, color="g", label="future_predict")
#plt.plot(range(len(true_data)-PLOT_RANGE-PREDICT_LEN,len(true_data)),true_data[-PLOT_RANGE-PREDICT_LEN:],color="r",alpha=0.5,label="true_data")
plt.legend()
plt.show()
