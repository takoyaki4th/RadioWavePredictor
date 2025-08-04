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


csv_path = f"{path}/result/WAVE0001/result_nd-001.csv" 
train_x,train_y=read_csv_and_convert_dataset(csv_path)


# モデル構築
print("🚀 新しいモデルを作成します")
model = Sequential()
model.add(Input(shape=(INPUT_LEN, FEATURES_NUM)))
model.add(LSTM(HIDDEN_NUM, return_sequences=True))
model.add(LSTM(HIDDEN_NUM // 2,return_sequences=False))
model.add(Dense(FEATURES_NUM))
model.add(Activation("linear"))
optimizer = Adam(learning_rate=0.005)
model.compile(loss="mean_squared_error", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)

history=model.fit(
    train_x,
    train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    #validation_data=(val_x,val_y),
    validation_split=0.1,
    callbacks=[early_stopping],
)

model.save(MODEL_PATH)
predicted=model.predict(train_x)

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
#plt.plot(range(0,len(predicted)), predicted, color="g", label="future_predict")
#plt.plot(range(0,len(data_numpy)),data_numpy,color="r",alpha=0.5,label="f")
plt.legend()
plt.show()
