import time 
import keras_tuner

from constant import *
from func import load_training_data
from hyperparam_tuning.build_model import build_model

def layer_unit_tuning():
    (train_x_arr,train_y_arr),(val_x,val_y)=load_training_data(TRAINING_COURCES,VALIDATION_COURCE,LEARN_MODE)

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        directory=path,
        project_name='tuner_result'
    )

    tuner.search(train_x_arr[0], train_y_arr[0], epochs=50, validation_data=(val_x, val_y))

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
