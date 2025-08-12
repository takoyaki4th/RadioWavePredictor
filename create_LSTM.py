import matplotlib.pyplot as plt
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from constant import *
from func import load_training_data

(train_x_arr,train_y_arr),(val_x,val_y)=load_training_data(TRAINING_COURCES,VALIDATION_COURCE,LEARN_MODE)

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

for i,cource in enumerate(TRAINING_COURCES):
    print("\n##########################")
    print(f"コース{cource}を学習します")
    print("##########################\n")
    history=model.fit(
        train_x_arr[i],
        train_y_arr[i],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x,val_y),
        callbacks=[early_stopping],
    )

model.save(MODEL_PATH)

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
