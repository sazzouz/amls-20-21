import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import (
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenetv2_preprocess_input,
)
from tensorflow.keras.applications.nasnet import (
    preprocess_input as nasnetmobile_preprocess_input,
)


def get_split_data(df, test_size=0.2, random_state=66):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def get_cnn_img_gen(is_cartoon=False):
    if not is_cartoon:
        train_img_gen = ImageDataGenerator(
            # Normalise images
            rescale=1.0 / 255,
            fill_mode="nearest",
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    else:
        train_img_gen = ImageDataGenerator(
            # Normalise images
            rescale=1.0 / 255,
            fill_mode="nearest",
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    test_img_gen = ImageDataGenerator(
        # Normalise images
        rescale=1.0 / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    return train_img_gen, test_img_gen


def get_mobilenet_img_gen(is_cartoon=False):
    if not is_cartoon:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=mobilenet_preprocess_input,
            fill_mode="nearest",
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    else:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=mobilenet_preprocess_input,
            fill_mode="nearest",
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    test_img_gen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess_input,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    return train_img_gen, test_img_gen


def get_mobilenetv2_img_gen(is_cartoon=False):
    if not is_cartoon:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=mobilenetv2_preprocess_input,
            fill_mode="nearest",
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    else:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=mobilenetv2_preprocess_input,
            fill_mode="nearest",
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    test_img_gen = ImageDataGenerator(
        preprocessing_function=mobilenetv2_preprocess_input,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    return train_img_gen, test_img_gen


def get_nasnetmobile_img_gen(is_cartoon=False):
    if not is_cartoon:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=nasnetmobile_preprocess_input,
            fill_mode="nearest",
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    else:
        train_img_gen = ImageDataGenerator(
            preprocessing_function=nasnetmobile_preprocess_input,
            fill_mode="nearest",
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.20,
        )
    test_img_gen = ImageDataGenerator(
        preprocessing_function=nasnetmobile_preprocess_input,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    return train_img_gen, test_img_gen


def get_img_data_from_directory(
    task_dir, train_img_gen, test_img_gen, target_size=(224, 224)
):
    train_flow = train_img_gen.flow_from_directory(
        os.path.join(task_dir, "train/"),
        class_mode="categorical",
        target_size=target_size,
        shuffle=True,
        subset="training",
    )
    val_flow = train_img_gen.flow_from_directory(
        os.path.join(task_dir, "train/"),
        class_mode="categorical",
        target_size=target_size,
        interpolation="nearest",
        shuffle=True,
        subset="validation",
    )
    test_flow = test_img_gen.flow_from_directory(
        os.path.join(task_dir, "test/"),
        class_mode="categorical",
        target_size=target_size,
        interpolation="nearest",
        shuffle=True,
    )
    return train_flow, val_flow, test_flow
