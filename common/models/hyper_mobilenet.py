from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    Flatten,
    Conv2D,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.mobilenet import (
    MobileNet,
    preprocess_input as mobilenet_preprocess_input,
)
from kerastuner import HyperModel


class HyperMobileNet(HyperModel):
    def __init__(self, num_classes, input_shape, binary_task, multi_task):
        self.input_shape = input_shape
        self.num_classes = num_classes
        if binary_task:
            self.output_activation = "sigmoid"
            self.model_loss = "binary_crossentropy"
        elif multi_task:
            self.output_activation = "softmax"
            self.model_loss = "categorical_crossentropy"

    def get_optimizer(self, hp, optimizer):
        lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
        if optimizer == "adam":
            return Adam(learning_rate=lr)
        elif optimizer == "sgd":
            return SGD(learning_rate=lr)
        elif optimizer == "rmsprop":
            return RMSprop(learning_rate=lr)

    def build(self, hp):
        ##### Setup hyperparameters
        optimizer = hp.Choice("optimizer", values=["adam", "sgd", "rmsprop"])
        first_dropout = hp.Float("first_dropout", 0.1, 0.6)
        second_dropout = hp.Float("second_dropout", 0.1, 0.6)
        first_dense_units = hp.Int(
            "first_dense_units", min_value=16, max_value=128, step=16
        )
        second_dense_units = hp.Int(
            "second_dense_units", min_value=16, max_value=128, step=16
        )
        ###### Construct model
        # Instantiate base model.
        base_model = MobileNet(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        # Freeze last 5 layers.
        for layer in base_model.layers[:-5]:
            layer.trainable = False
        # Instantiate head model .
        head_model = base_model.output
        head_model = GlobalAveragePooling2D()(head_model)
        head_model = Dropout(rate=first_dropout)(head_model)
        head_model = Dense(units=first_dense_units, activation="relu")(head_model)
        head_model = Dropout(rate=second_dropout)(head_model)
        head_model = Dense(units=second_dense_units, activation="relu")(head_model)
        output = Dense(self.num_classes, activation=self.output_activation)(head_model)
        model = Model(base_model.input, output)
        model.compile(
            optimizer=self.get_optimizer(hp, optimizer),
            loss=self.model_loss,
            metrics=["accuracy"],
        )
        return model