"""
Model specification.
"""
import os
from .preprocessing import get_input_shape
from .tuning import tune_model, schedule
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard,
    TerminateOnNaN,
    ProgbarLogger,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16


def get_model(cartoon_set_img_dir):
    """
    Model Architecture For B2
    """
    # model = Sequential()
    # # First filter
    # # Use 16 3x3 kernels with ReLU after.
    # model.add(
    #     Conv2D(
    #         16,
    #         3,
    #         padding="same",
    #         activation="relu",
    #         input_shape=get_input_shape(cartoon_set_img_dir, channels=3),
    #     )
    # )
    # # Pooling layer
    # model.add(MaxPooling2D())
    # # Dropout layer
    # model.add(Dropout(0.3))
    # # Use 32 3x3 kernels with ReLU after. Notice this is double the last layer.
    # # Second filter
    # model.add(Conv2D(32, 3, padding="same", activation="relu"))
    # # Pooling layer
    # model.add(MaxPooling2D())
    # # Dropout layer
    # # model.add(Dropout(0.3))
    # # Use 64 3x3 kernels with ReLU after. Notice this is double the last layer.
    # # Third filter
    # # model.add(Conv2D(64, 3, padding="same", activation="relu"))
    # # Pooling layer
    # # model.add(MaxPooling2D())
    # # Flatten for use with fully-connected layers
    # # Dropout layer
    # # model.add(Dropout(0.3))
    # # Flatten layer
    # # Fourth Filter
    # model.add(Flatten())
    # # Fully connected layer with 512 neurons
    # # model.add(Dense(512, activation="relu"))
    # # Dropout layer
    # # model.add(Dropout(0.3))
    # # Output layer
    # model.add(Dense(2, activation="softmax"))
    # # Comile the model with parameters

    num_start_filters = 16
    kernel_size = 3
    height = 218
    width = 178
    fcl_size = 512
    num_classes = 6

    model = Sequential()

    # Add 1st convolution block
    model.add(
        Conv2D(
            filters=16,
            input_shape=get_input_shape(cartoon_set_img_dir),
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Add 2nd convolution block
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Add 3rd convolution block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Add 4th convolution block
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Output Layer (5 clases for face shapes)
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    opt = Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # model.compile(
    #     optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
    # )

    model_summary = model.summary()
    return model, model_summary


def train_model(b2_model, epochs, eye_color_train_flow, eye_color_val_flow, b2_dir):
    eye_color_train_flow.reset()
    # Setup step parameters for training method.
    train_step_size = eye_color_train_flow.samples // eye_color_train_flow.batch_size
    val_step_size = eye_color_val_flow.samples // eye_color_val_flow.batch_size

    # Setup callback functions for training method.

    # Custom learning-rate scheduler imported from tuning module.
    lr_sched_cb = LearningRateScheduler(schedule, verbose=0)

    # Stop training once converged to best model
    early_stop_cb = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=2,
        verbose=1,
        mode="auto",
        min_lr=0.00001,
    )
    csv_logger_cb = CSVLogger(os.path.join(b2_dir, "logs/b2_training.log"))
    tensorboard_cb = TensorBoard(
        log_dir=os.path.join(b2_dir, "./logs"), write_graph=True, write_images=True
    )
    term_nan_cb = TerminateOnNaN()
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(b2_dir, "b2_best_model.h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    # Train model
    eye_color_history = b2_model.fit(
        x=eye_color_train_flow,
        epochs=epochs,
        steps_per_epoch=train_step_size,
        validation_data=eye_color_val_flow,
        validation_steps=val_step_size,
        callbacks=[
            early_stop_cb,
            reduce_lr_cb,
            csv_logger_cb,
            checkpoint_cb,
            tensorboard_cb,
            term_nan_cb,
        ],
        verbose=1,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )
    return eye_color_history


def evaluate_model(model, eye_color_val_flow, b2_dir):
    eye_color_val_flow.reset()
    val_step_size = eye_color_val_flow.samples // eye_color_val_flow.batch_size
    model.load_weights(os.path.join(b2_dir, "b2_best_model.h5"))
    eval_score = model.evaluate(
        eye_color_val_flow, steps=val_step_size, return_dict=True
    )
    return eval_score


def predict_model(model, eye_color_test_flow, eye_color_test_df, b2_dir):
    # Return predictions for entire dataset
    eye_color_test_flow.reset()
    test_step_size = eye_color_test_flow.samples // eye_color_test_flow.batch_size
    model.load_weights(os.path.join(b2_dir, "b2_best_model.h5"))
    eye_color_test_predict_df = eye_color_test_df.copy()
    predictions = model.predict(eye_color_test_flow, steps=test_step_size)
    predictions_classes = np.argmax(predictions, axis=1)
    test_classes = eye_color_test_flow.classes
    # TODO: Needs to be the length of the resulting dataframe following removed items
    # eye_color_test_predict_df["prediction"] = predictions

    return predictions_classes, test_classes, eye_color_test_predict_df
