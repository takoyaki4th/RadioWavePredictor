import os

COURCE_NUM = 4

path = os.path.dirname(__file__)
MODEL_PATH = path+f"/cource_{COURCE_NUM}_model_50.h5"
CSV_PATH = f"{path}/result/WAVE000{COURCE_NUM}/result_nt-001.csv" 

INPUT_LEN = 50
HIDDEN_NUM = 128
FEATURES_NUM = 1 #特徴量の数
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.002

PREDICT_LEN = 1000
PLOT_RANGE = 300