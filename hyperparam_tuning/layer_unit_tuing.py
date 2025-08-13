import keras_tuner
from keras.callbacks import EarlyStopping

from setting import *
from func import load_training_data
from hyperparam_tuning.build_model import build_model

# プログラムの実行はルートディレクトリにあるhyperparam.pyから行ってください
def layer_unit_tuning():
    train_dataset_arr,validation_dataset=load_training_data(TRAINING_COURCES,VALIDATION_COURCE,LEARN_MODE)

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        directory=path,
        project_name='tuner_result'
    )

    tuner.search(
        train_dataset_arr[0],
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=20)],
    )

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"一番良かったパラメータ:{best_hps.values}")
