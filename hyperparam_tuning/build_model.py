import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.optimizers import Adam

from constant import *

def make_model(units=4, depth=2):
    print("🚀 新しいモデルを作成します")
    model = Sequential()
    model.add(Input(shape=(INPUT_LEN, FEATURES_NUM)))
    for _ in range(depth-1):
        model.add(SimpleRNN(units, return_sequences=True))
    model.add(SimpleRNN(units, return_sequences=False))
    model.add(Dense(FEATURES_NUM))
    model.add(Activation("linear"))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model

def build_model(hp):
    units = hp.Choice('units', [ 8, 16, 32,64])
    depth = hp.Choice('depth', [1, 2, 3, 4])
    model = make_model(units, depth)
    return model

