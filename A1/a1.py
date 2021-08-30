"""
A1 Task Orchestration.
"""
# Import library modules
import os
import sys
import shutil
from tensorflow.keras.models import load_model

# Import common modules
from common.models.hyper_cnn import HyperCNN
from common.models.hyper_mobilenet import HyperMobileNet
from common.models.hyper_mobilenetv2 import HyperMobileNetV2
from common.models.hyper_nasnetmobile import HyperNASNetMobile
from common.preprocessing import (
    get_split_data,
    get_cnn_img_gen,
    get_mobilenet_img_gen,
    get_mobilenetv2_img_gen,
    get_nasnetmobile_img_gen,
    get_img_data_from_directory,
)
from common.activities import tune, train, test
from common.utils import get_callbacks, cleanup, cast_to_list
from common.plotting import (
    plot_all_accuracy,
    plot_single_accuracy_loss,
    plot_all_loss
)

# Import local modules
from .preparation import load_data, process_data

class_mapper_dict = {"-1": "female", "1": "male"}


# Task class including methods for each distinct model
class A1:
    def prepare(self, a1_dir, celeba_img_dir, celeba_df):
        # Get train / test split for gender-labelled images
        train_df, test_df = load_data(celeba_df)
        print("\n>>> Generating Training Dataset In Temporary Directory\n")
        train_valid, train_invalid = process_data(
            a1_dir,
            celeba_img_dir,
            train_df,
            "train",
            "gender",
            class_mapper_dict,
        )
        print("\n>>> Generating Testing Dataset In Temporary Directory\n")
        test_valid, test_invalid = process_data(
            a1_dir,
            celeba_img_dir,
            test_df,
            "test",
            "gender",
            class_mapper_dict,
        )

    def cnn(self, a1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_cnn_img_gen()
        train_data, val_data, test_data = get_img_data_from_directory(
            a1_dir, train_img_gen, test_img_gen
        )
        cnn = HyperCNN(
            num_classes=2,
            filters=16,
            kernel_size=(3, 3),
            input_shape=(224, 224, 3),
            binary_task=True,
            multi_task=False,
        )
        best_model = tune(a1_dir, "cnn_tuning", cnn, train_data, val_data)
        best_model.save(
            os.path.join(a1_dir, "saved_models", "cnn_best_model_tuned.h5"),
            overwrite=True,
        )
        best_model_tuned = load_model(os.path.join(a1_dir, "saved_models", "cnn_best_model_tuned.h5"))
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(a1_dir, 'A1', 'cnn_tuned')
        )
        best_model_trained.save(
            os.path.join(a1_dir, "saved_models", "cnn_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(a1_dir, "saved_models", "cnn_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(a1_dir, history, 'A1', model_name='CNN')
        return history_obj, results_obj

    def mobilenet(self, a1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen()
        train_data, val_data, test_data = get_img_data_from_directory(
            a1_dir, train_img_gen, test_img_gen
        )
        mobilenet = HyperMobileNet(
            num_classes=2, input_shape=(224, 224, 3), binary_task=True, multi_task=False
        )
        best_model = tune(a1_dir, "mobilenet_tuning", mobilenet, train_data, val_data)
        best_model.save(
            os.path.join(a1_dir, "saved_models", "mobilenet_best_model_tuned.h5"),
            overwrite=True,
        )
        best_model_tuned = load_model(
            os.path.join(a1_dir, "saved_models", "mobilenet_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(a1_dir, 'A1', 'mobilenet_tuned')
        )
        best_model_trained.save(
            os.path.join(a1_dir, "saved_models", "mobilenet_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(a1_dir, "saved_models", "mobilenet_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(a1_dir, history, 'A1', model_name='MobileNet')
        return history_obj, results_obj

    def mobilenetv2(self, a1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen()
        train_data, val_data, test_data = get_img_data_from_directory(
            a1_dir, train_img_gen, test_img_gen
        )
        mobilenetv2 = HyperMobileNetV2(
            num_classes=2, input_shape=(224, 224, 3), binary_task=True, multi_task=False
        )
        best_model = tune(
            a1_dir,
            "mobilenetv2_tuning",
            mobilenetv2,
            train_data,
            val_data,
        )
        best_model.save(
            os.path.join(a1_dir, "saved_models", "mobilenetv2_best_model_tuned.h5"),
            overwrite=True,
        )
        best_model_tuned = load_model(
            os.path.join(a1_dir, "saved_models", "mobilenetv2_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(a1_dir, 'A1', 'mobilenetv2_tuned')
        )
        best_model_trained.save(
            os.path.join(a1_dir, "saved_models", "mobilenetv2_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(a1_dir, "saved_models", "mobilenetv2_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(a1_dir, history, 'A1', model_name='MobileNetV2')
        return history_obj, results_obj

    def nasnetmobile(self, a1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen()
        train_data, val_data, test_data = get_img_data_from_directory(
            a1_dir, train_img_gen, test_img_gen
        )
        nasnetmobile = HyperNASNetMobile(
            num_classes=2, input_shape=(224, 224, 3), binary_task=True, multi_task=False
        )
        best_model = tune(
            a1_dir, "nasnetmobile_tuning", nasnetmobile, train_data, val_data
        )
        best_model.save(
            os.path.join(a1_dir, "saved_models", "nasnetmobile_best_model_tuned.h5"),
            overwrite=True,
        )
        best_model_tuned = load_model(
            os.path.join(a1_dir, "saved_models", "nasnetmobile_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(a1_dir, 'A1', 'nasnetmobile_tuned')
        )
        best_model_trained.save(
            os.path.join(a1_dir, "saved_models", "nasnetmobile_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(a1_dir, "saved_models", "nasnetmobile_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(a1_dir, history, 'A1', model_name='NASNetMobile')
        return history_obj, results_obj

    def analyse(self, a1_dir, all_a1_histories, all_a1_results):
        plot_all_accuracy(a1_dir, all_a1_histories, 'A1')
        plot_all_loss(a1_dir, all_a1_histories, 'A1')
