import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from constant import *
from func import *

train_x_arr=[]
train_y_arr=[]

for i in range(1,4):
    csv_path = f"{path}/result/WAVE00{i:02d}/result_nt-001.csv" 
    train_x_i,train_y_i=read_csv_and_convert_dataset(csv_path)

    train_x_arr.append(train_x_i)
    train_y_arr.append(train_y_i)

csv_path = f"{path}/result/WAVE0004/result_nt-001.csv" 
val_x,val_y=read_csv_and_convert_dataset(csv_path)

# モデル構築
print("🚀 新しいモデルを作成します")
model = Sequential()
model.add(Input(shape=(INPUT_LEN, FEATURES_NUM)))
model.add(LSTM(HIDDEN_NUM, return_sequences=True))
model.add(LSTM(HIDDEN_NUM // 2,return_sequences=False))
model.add(Dense(FEATURES_NUM))
model.add(Activation("linear"))
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)

for i in range(len(train_x_arr)):
    print("######################")
    print(f"現在の学習コース:{i+1}")
    print("######################")
    history=model.fit(
        train_x_arr[i],
        train_y_arr[i],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x,val_y),
        callbacks=[early_stopping],
    )

model.save(MODEL_PATH)
#predicted=model.predict(train_x_arr[0])

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
#plt.plot(range(0,len(predicted)), predicted, color="g", label="future_predict")
#plt.plot(range(0,len(data_numpy)),data_numpy,color="r",alpha=0.5,label="f")
plt.legend()
plt.show()
