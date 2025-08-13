import matplotlib.pyplot as plt
import time 
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from setting import *
from func import load_training_data

#コード実行時間計測
start_time=time.time()

train_dataset_arr,val_dataset=load_training_data(TRAINING_COURCES,VALIDATION_COURCE,LEARN_MODE)

# モデル構築
print("🚀 新しいモデルを作成します")
model = Sequential()
model.add(Input(shape=(INPUT_LEN, FEATURES_NUM)))
model.add(SimpleRNN(HIDDEN_NUM, return_sequences=True))
model.add(SimpleRNN(HIDDEN_NUM, return_sequences=True))
model.add(SimpleRNN(HIDDEN_NUM, return_sequences=True))
model.add(SimpleRNN(HIDDEN_NUM, return_sequences=False))
model.add(Dense(FEATURES_NUM))
model.add(Activation("linear"))
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)

for i,cource in enumerate(TRAINING_COURCES):
    print("\n##########################")
    print(f"コース{cource}を学習します")
    print("##########################\n")
    history=model.fit(
        train_dataset_arr[i],
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[early_stopping],
    )

model.save(MODEL_PATH)

end_time=time.time()
print(f"実行時間:{(end_time-start_time):2f}秒")

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
