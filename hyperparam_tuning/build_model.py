'''
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.optimizers import Adam

from setting import *
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
'''
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras_tuner import HyperModel

from setting import *

class RNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(INPUT_LEN, 1)))

        depth=hp.Int("depth",1,4)
        # 層の数・サイズ
        for i in range(depth):
            model.add(USE_RNN_LAYER(units=hp.Int(f"units_{i}", 4, 256, step=4),return_sequences=(i < depth - 1)))

        optimizer = Adam(learning_rate=hp.Int(f"lr",1e-4,1e-3))
        model.add(Dense(1, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=optimizer,
        )
        model.summary()
        return model
