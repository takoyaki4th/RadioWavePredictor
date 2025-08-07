import os
path = os.path.dirname(__file__)

TRAINING_COURCES=range(1,4) #学習するコース番号群
VALIDATION_COURCE=5 #検証に使うコース番号
LEARN_MODE="t" #学習データの種類 tなら時間、dなら距離

### 学習モデルに関する設定 ### 
INPUT_LEN = 50
HIDDEN_NUM = 128
FEATURES_NUM = 1 #特徴量の数
BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.002

#作成するモデルのパスと名前(予測でもこれを参照しています)
MODEL_PATH = path+f"/{LEARN_MODE}_{INPUT_LEN}_{HIDDEN_NUM}.h5"

### 予測に関する設定 ###
PREDICT_COURCE=4 #予測したいコース番号
PREDICT_LEN = 1000 #再帰で予測する長さ
PLOT_RANGE = 300 #グラフとして表示する範囲