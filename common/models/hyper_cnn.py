from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.mobilenet import (
    MobileNet,
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenetv2_preprocess_input,
)
from tensorflow.keras.applications.nasnet import (
    NASNetMobile,
    preprocess_input as nasnetmobile_preprocess_input,
)
from kerastuner.tuners import Hyperband
import kerastuner as kt


class HyperCNN(kt.HyperModel):
    def __init__(
        self, num_classes, filters, kernel_size, input_shape, binary_task, multi_task
    ):
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        if binary_task:
            self.output_activation = "sigmoid"
            self.model_loss = "binary_crossentropy"
        elif multi_task:
            self.output_activation = "softmax"
            self.model_loss = "categorical_crossentropy"

    def get_optimizer(self, hp, optimizer):
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        if optimizer == "adam":
            return Adam(learning_rate=lr)
        elif optimizer == "sgd":
            return SGD(learning_rate=lr)
        elif optimizer == "rmsprop":
            return RMSprop(learning_rate=lr)

    def build(self, hp):
        ###### Setup hyperparamaters
        pooling_1 = hp.Choice("pooling_1", ["avg", "max"])
        pooling_2 = hp.Choice("pooling_2", ["avg", "max"])
        pooling_3 = hp.Choice("pooling_3", ["avg", "max"])
        dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
        dropout = hp.Float("dropout", 0.1, 0.6)
        optimizer = hp.Choice("optimizer", values=["adam", "sgd", "rmsprop"])

        ###### Construct model
        # Instantiate sequential model.
        model = Sequential()
        # Use 16 3x3 kernels with ReLU after.
        model.add(
            Conv2D(
                self.filters,
                self.kernel_size,
                padding="same",
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        # Pooling layer
        if pooling_1 == "max":
            model.add(MaxPool2D())
        else:
            model.add(AvgPool2D())
        # Use 32 3x3 kernels with ReLU after. Notice this is double the last layer.
        model.add(
            Conv2D(
                self.filters * 2, self.kernel_size, padding="same", activation="relu"
            )
        )
        # Pooling layer
        if pooling_2 == "max":
            model.add(MaxPool2D())
        else:
            model.add(AvgPool2D())
        # Use 64 3x3 kernels with ReLU after. Notice this is double the last layer.
        model.add(
            Conv2D(
                self.filters * 4, self.kernel_size, padding="same", activation="relu"
            )
        )
        # Pooling layer
        if pooling_3 == "max":
            model.add(MaxPool2D())
        else:
            model.add(AvgPool2D())
        # Flatten for use with fully-connected layers
        model.add(Flatten())
        # Fully connected layer with 512 neurons
        model.add(Dense(dense_units, activation="relu"))
        # Regularization layer using dropout
        model.add(Dropout(dropout))
        # Output layer
        model.add(Dense(self.num_classes, activation=self.output_activation))
        # Compile model
        model.compile(
            optimizer=self.get_optimizer(hp, optimizer),
            loss=self.model_loss,
            metrics=["accuracy"],
        )
        return model
