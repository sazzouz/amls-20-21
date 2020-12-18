import os
import shutil
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau,
)
from tensorflow.keras.backend import clear_session


def get_callbacks(dir, task_name, model_name):
    print("\nGetting callbacks\n")
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(
                dir,
                "logs",
                str(task_name),
                "checkpoints",
                str(model_name),
                ".{epoch:02d}-{val_accuracy:.2f}.h5",
            ),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        CSVLogger(
            filename=os.path.join(
                dir,
                'logs', 
                '{0}-{1}.log'.format(task_name, model_name) 
            ),
            separator=',',
            append=False
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.2, patience=2, min_lr=0.001, verbose=1
        ),
    ]
    return callbacks


def cleanup(a1_dir):
    print("\n>>> Clearing Keras Session\n")
    clear_session()
    # print("\n>>> Removing Temporary Train/Test Image Directories\n")
    # for folder in ["train", "test"]:
    #     if os.path.exists(os.path.join(a1_dir, folder)) and os.path.isdir(
    #         os.path.join(a1_dir, folder)
    #     ):
    #         try:
    #             shutil.rmtree(os.path.join(a1_dir, folder))
    #         except OSError:
    #             print("Failed To Remove Directory %s" % folder)

def cast_to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]