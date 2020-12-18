"""
B1 Task Orchestration.
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

class_mapper_dict = {
    "0": "shape0",
    "1": "shape1",
    "2": "shape2",
    "3": "shape3",
    "4": "shape4",
}

class B1:
    def prepare(self, b1_dir, celeba_img_dir, celeba_df):
        # Get train / test split for face_shape-labelled images
        train_df, test_df = load_data(celeba_df)
        print("\n>>> Generating Training Dataset In Temporary Directory\n")
        process_data(
            b1_dir,
            celeba_img_dir,
            train_df,
            "train",
            "face_shape",
            class_mapper_dict,
        )
        # plot_valid_invalid(train_valid, train_invalid)
        print("\n>>> Generating Testing Dataset In Temporary Directory\n")
        process_data(
            b1_dir,
            celeba_img_dir,
            test_df,
            "test",
            "face_shape",
            class_mapper_dict,
        )
        # plot_valid_invalid(test_valid, test_invalid)

    def cnn(self, b1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_cnn_img_gen(is_cartoon=True)
        train_data, val_data, test_data = get_img_data_from_directory(
            b1_dir, train_img_gen, test_img_gen
        )
        cnn = HyperCNN(
            num_classes=5,
            filters=16,
            kernel_size=(3, 3),
            input_shape=(224, 224, 3),
            binary_task=False,
            multi_task=True,
        )
        # best_model = tune(b1_dir, "cnn_tuning", cnn, train_data, val_data)
        # best_model.save(
        #     os.path.join(b1_dir, "saved_models", "cnn_best_model_tuned.h5"),
        #     overwrite=True,
        # )
        best_model_tuned = load_model(os.path.join(b1_dir, "saved_models", "cnn_best_model_tuned.h5"))
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(b1_dir, 'B1', 'cnn_tuned')
        )
        best_model_trained.save(
            os.path.join(b1_dir, "saved_models", "cnn_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(b1_dir, "saved_models", "cnn_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(b1_dir, history, 'B1', model_name='CNN')
        return history_obj, results_obj

    def mobilenet(self, b1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen(is_cartoon=True)
        train_data, val_data, test_data = get_img_data_from_directory(
            b1_dir, train_img_gen, test_img_gen
        )
        mobilenet = HyperMobileNet(
            num_classes=5, input_shape=(224, 224, 3), binary_task=False, multi_task=True
        )
        # best_model = tune(b1_dir, "mobilenet_tuning", mobilenet, train_data, val_data)
        # best_model.save(
        #     os.path.join(b1_dir, "saved_models", "mobilenet_best_model_tuned.h5"),
        #     overwrite=True,
        # )
        best_model_tuned = load_model(
            os.path.join(b1_dir, "saved_models", "mobilenet_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(b1_dir, 'B1', 'mobilenet_tuned')
        )
        best_model_trained.save(
            os.path.join(b1_dir, "saved_models", "mobilenet_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(b1_dir, "saved_models", "mobilenet_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(b1_dir, history, 'B1', model_name='MobileNet')
        return history_obj, results_obj

    def mobilenetv2(self, b1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen(is_cartoon=True)
        train_data, val_data, test_data = get_img_data_from_directory(
            b1_dir, train_img_gen, test_img_gen
        )
        mobilenetv2 = HyperMobileNetV2(
            num_classes=5, input_shape=(224, 224, 3), binary_task=False, multi_task=True
        )
        # best_model = tune(
        #     b1_dir,
        #     "mobilenetv2_tuning",
        #     mobilenetv2,
        #     train_data,
        #     val_data,
        # )
        # best_model.save(
        #     os.path.join(b1_dir, "saved_models", "mobilenetv2_best_model_tuned.h5"),
        #     overwrite=True,
        # )
        best_model_tuned = load_model(
            os.path.join(b1_dir, "saved_models", "mobilenetv2_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(b1_dir, 'B1', 'mobilenetv2_tuned')
        )
        best_model_trained.save(
            os.path.join(b1_dir, "saved_models", "mobilenetv2_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(b1_dir, "saved_models", "mobilenetv2_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(b1_dir, history, 'B1', model_name='MobileNetV2')
        return history_obj, results_obj

    def nasnetmobile(self, b1_dir):
        history_obj = {}
        train_img_gen, test_img_gen = get_mobilenet_img_gen(is_cartoon=True)
        train_data, val_data, test_data = get_img_data_from_directory(
            b1_dir, train_img_gen, test_img_gen
        )
        nasnetmobile = HyperNASNetMobile(
            num_classes=5, input_shape=(224, 224, 3), binary_task=False, multi_task=True
        )
        # best_model = tune(
        #     b1_dir, "nasnetmobile_tuning", nasnetmobile, train_data, val_data
        # )
        # best_model.save(
        #     os.path.join(b1_dir, "saved_models", "nasnetmobile_best_model_tuned.h5"),
        #     overwrite=True,
        # )
        best_model_tuned = load_model(
            os.path.join(b1_dir, "saved_models", "nasnetmobile_best_model_tuned.h5")
        )
        history, best_model_trained = train(
            best_model_tuned, train_data, val_data, get_callbacks(b1_dir, 'B1', 'nasnetmobile_tuned')
        )
        best_model_trained.save(
            os.path.join(b1_dir, "saved_models", "nasnetmobile_best_model_trained.h5")
        )
        best_model_trained = load_model(
            os.path.join(b1_dir, "saved_models", "nasnetmobile_best_model_trained.h5")
        )
        results_obj = test(best_model_trained, test_data)
        history_obj["loss"] = history.history["loss"]
        history_obj["val_loss"] = history.history["val_loss"]
        history_obj["accuracy"] = history.history["accuracy"]
        history_obj["val_accuracy"] = history.history["val_accuracy"]
        plot_single_accuracy_loss(b1_dir, history, 'B1', model_name='NASNetMobile')
        return history_obj, results_obj

    def analyse(self, b1_dir, all_b1_histories, all_b1_results):
        plot_all_accuracy(b1_dir, all_b1_histories, 'B1')
        plot_all_loss(b1_dir, all_b1_histories, 'B1')
