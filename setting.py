import os
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop
path = os.path.dirname(__file__)

TRAINING_COURCES=[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17] #学習するコース番号群
VALIDATION_COURCES=[5] #検証に使うコース番号群
LEARN_MODE="t" #学習データの種類 tなら時間、dなら距離

### 学習モデルに関する設定 ### 
USE_RNN_LAYER = SimpleRNN #使用するRNNの種類、layerを作るときに使用するclassを直接指定する
USE_OPTIMIZER = AdamW
INPUT_LEN = 100
HIDDEN_NUMS = [8] #隠れ層のユニット数を配列で指定
IN_FEATURES_NUM = 2 #入力特徴量の数
OUT_FEATURES_NUM = 1 
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 0.00025

#作成するモデルのパスと名前(予測でもこれを参照しています)
MODEL_PATH = path+f"/{USE_RNN_LAYER.__name__}_"+"_".join(map(str, HIDDEN_NUMS))+".keras"

### 予測に関する設定 ###
PREDICT_COURCE=4 #予測したいコース番号
PREDICT_LEN = 1000 #再帰で予測する長さ
PLOT_START = 2000
PLOT_RANGE = 500 #グラフとして表示する範囲
