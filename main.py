# Import dependancies for main file.
import os
import sys
import pandas as pd
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from prettytable import PrettyTable

# Import class defintions for each task.
from A1 import a1
from A2 import a2
from B1 import b1
from B2 import b2

# Import utils
from common.utils import cleanup

print("\n~~~~~ Setup ~~~~~\n")

# Setup directory refrerencing
base_dir = "Datasets/"
a1_dir = "A1/"
a2_dir = "A2/"
b1_dir = "B1/"
b2_dir = "B2/"
celeba_dir = os.path.join(base_dir, "celeba/")
celeba_test_dir = os.path.join(base_dir, "celeba_test/")
cartoon_set_dir = os.path.join(base_dir, "cartoon_set/")
cartoon_set_test_dir = os.path.join(base_dir, "cartoon_set_test/")
celeba_img_dir = os.path.join(celeba_dir, "img/")
celeba_labels_dir = os.path.join(celeba_dir, "labels.csv")
celeba_test_labels_dir = os.path.join(celeba_test_dir, "labels.csv")
celeba_test_img_dir = os.path.join(celeba_test_dir, "img/")
cartoon_set_img_dir = os.path.join(cartoon_set_dir, "img/")
cartoon_set_labels_dir = os.path.join(cartoon_set_dir, "labels.csv")
cartoon_set_test_labels_dir = os.path.join(cartoon_set_test_dir, "labels.csv")
cartoon_set_test_img_dir = os.path.join(cartoon_set_test_dir, "img/")

# Instantiate shared dataset references
celeba_df = pd.read_csv(celeba_labels_dir, sep="\t", dtype=str)
cartoon_set_df = pd.read_csv(cartoon_set_labels_dir, sep="\t", dtype=str)

# Instantiate configuration params
epochs = 10
history_dict = {}
results_dict = {}

# Specify results table headings for use with PrettyTable package
results_table = PrettyTable()
results_table.field_names = [
    "Task",
    "Activity",
    "CNN",
    "MobileNet",
    "MobileNetV2",
    "NASNetMobile",
]

# ======================================================================================================================
# Task A1
print("\n~~~~~ Task A1 ~~~~~\n")
# Add new keys for this task
history_dict["a1"] = {}
results_dict["a1"] = {}
# Instantiate A1 task class
a1 = a1.A1()
# Data preprocessing for task A1
print("\n> Generating A1 Datasets\n")
a1.prepare(a1_dir, celeba_img_dir, celeba_df)
print("\n>>> A1 CNN \n")
history_dict["a1"]["cnn"] = {}
results_dict["a1"]["cnn"] = {}
cnn_history_obj, cnn_results_obj = a1.cnn(a1_dir)
history_dict["a1"]["cnn"] = cnn_history_obj
results_dict["a1"]["cnn"] = cnn_results_obj
print("\n>>> A1 MobileNet \n")
history_dict["a1"]["mobilenet"] = {}
results_dict["a1"]["mobilenet"] = {}
mobilenet_history_obj, mobilenet_results_obj = a1.mobilenet(a1_dir)
history_dict["a1"]["mobilenet"] = mobilenet_history_obj
results_dict["a1"]["mobilenet"] = mobilenet_results_obj
print("\n>>> A1 MobileNetV2.\n")
history_dict["a1"]["mobilenetv2"] = {}
results_dict["a1"]["mobilenetv2"] = {}
mobilenetv2_history, mobilenetv2_results = a1.mobilenetv2(a1_dir)
history_dict["a1"]["mobilenetv2"] = mobilenetv2_history
results_dict["a1"]["mobilenetv2"] = mobilenetv2_results
print("\n>>> A1 NASNetMobile \n")
history_dict["a1"]["nasnetmobile"] = {}
results_dict["a1"]["nasnetmobile"] = {}
nasnetmobile_history, nasnetmobile_results = a1.nasnetmobile(a1_dir)
history_dict["a1"]["nasnetmobile"] = nasnetmobile_history
results_dict["a1"]["nasnetmobile"] = nasnetmobile_results

results_table.add_rows(
    [
        [
            "A1",
            "Training Acc",
            max(history_dict["a1"]["cnn"]["accuracy"]),
            max(history_dict["a1"]["mobilenet"]["accuracy"]),
            max(history_dict["a1"]["mobilenetv2"]["accuracy"]),
            max(history_dict["a1"]["nasnetmobile"]["accuracy"]),
        ],
        [
            "A1",
            "Validation Acc",
            max(history_dict["a1"]["cnn"]["val_accuracy"]),
            max(history_dict["a1"]["mobilenet"]["val_accuracy"]),
            max(history_dict["a1"]["mobilenetv2"]["val_accuracy"]),
            max(history_dict["a1"]["nasnetmobile"]["val_accuracy"]),
        ],
        [
            "A1",
            "Testing Acc",
            results_dict["a1"]["cnn"]["accuracy"],
            results_dict["a1"]["mobilenet"]["accuracy"],
            results_dict["a1"]["mobilenetv2"]["accuracy"],
            results_dict["a1"]["nasnetmobile"]["accuracy"],
        ],
    ]
)
print("\n>>> A1 Analysis \n")
all_a1_histories = [cnn_history_obj, mobilenet_history_obj, mobilenetv2_history, nasnetmobile_history]
all_a1_results = [cnn_results_obj, mobilenet_results_obj, mobilenetv2_results, nasnetmobile_results]
a1.analyse(a1_dir, all_a1_histories, all_a1_results)
print("\n>>> Cleaning Up A1 Task.\n")
cleanup(a1_dir)
print("Clean up memory/GPU etc...")  # TODO: Some code to free memory if necessary.
print('\n\n\n>>> LATEST RESULTS: \n\n\n')
print(results_table)

# ======================================================================================================================
# Task A2
print("\n~~~~~ Task A2 ~~~~~\n")
history_dict["a2"] = {}
results_dict["a2"] = {}
# Instantiate A2 task class
a2 = a2.A2()
# Data preprocessing for task A2
print("\n> Generating A2 Datasets\n")
a2.prepare(a2_dir, celeba_img_dir, celeba_df)
print("\n>>> A2 CNN \n")
history_dict["a2"]["cnn"] = {}
results_dict["a2"]["cnn"] = {}
cnn_history_obj, cnn_results_obj = a2.cnn(a2_dir)
history_dict["a2"]["cnn"] = cnn_history_obj
results_dict["a2"]["cnn"] = cnn_results_obj
print("\n>>> A2 MobileNet \n")
history_dict["a2"]["mobilenet"] = {}
results_dict["a2"]["mobilenet"] = {}
mobilenet_history_obj, mobilenet_results_obj = a2.mobilenet(a2_dir)
history_dict["a2"]["mobilenet"] = mobilenet_history_obj
results_dict["a2"]["mobilenet"] = mobilenet_results_obj
print("\n>>> A2 MobileNetV2.\n")
history_dict["a2"]["mobilenetv2"] = {}
results_dict["a2"]["mobilenetv2"] = {}
mobilenetv2_history, mobilenetv2_results = a2.mobilenetv2(a2_dir)
history_dict["a2"]["mobilenetv2"] = mobilenetv2_history
results_dict["a2"]["mobilenetv2"] = mobilenetv2_results
print("\n>>> A2 NASNetMobile \n")
history_dict["a2"]["nasnetmobile"] = {}
results_dict["a2"]["nasnetmobile"] = {}
nasnetmobile_history, nasnetmobile_results = a2.nasnetmobile(a2_dir)
history_dict["a2"]["nasnetmobile"] = nasnetmobile_history
results_dict["a2"]["nasnetmobile"] = nasnetmobile_results

results_table.add_rows(
    [
        [
            "A2",
            "Training Acc",
            max(history_dict["a2"]["cnn"]["accuracy"]),
            max(history_dict["a2"]["mobilenet"]["accuracy"]),
            max(history_dict["a2"]["mobilenetv2"]["accuracy"]),
            max(history_dict["a2"]["nasnetmobile"]["accuracy"]),
        ],
        [
            "A2",
            "Validation Acc",
            max(history_dict["a2"]["cnn"]["val_accuracy"]),
            max(history_dict["a2"]["mobilenet"]["val_accuracy"]),
            max(history_dict["a2"]["mobilenetv2"]["val_accuracy"]),
            max(history_dict["a2"]["nasnetmobile"]["val_accuracy"]),
        ],
        [
            "A2",
            "Testing Acc",
            results_dict["a2"]["cnn"]["accuracy"],
            results_dict["a2"]["mobilenet"]["accuracy"],
            results_dict["a2"]["mobilenetv2"]["accuracy"],
            results_dict["a2"]["nasnetmobile"]["accuracy"],
        ],
    ]
)
print("\n>>> A2 Analysis \n")
all_a2_histories = [cnn_history_obj, mobilenet_history_obj, mobilenetv2_history, nasnetmobile_history]
all_a2_results = [cnn_results_obj, mobilenet_results_obj, mobilenetv2_results, nasnetmobile_results]
a2.analyse(a2_dir, all_a2_histories, all_a2_results)
print("\n>>> Cleaning Up A2 Task.\n")
cleanup(a2_dir)
print("Clean up memory/GPU etc...")  # TODO: Some code to free memory if necessary.
print('\n\n\n>>> LATEST RESULTS: \n\n\n')
print(results_table)


# ======================================================================================================================
# Task B1
print("\n~~~~~ Task B1 ~~~~~\n")
history_dict["b1"] = {}
results_dict["b1"] = {}
# Instantiate B1 task class
b1 = b1.B1()
# Data preprocessing for task B1
print("\n> Generating B1 Datasets\n")
b1.prepare(b1_dir, cartoon_set_img_dir, cartoon_set_df)
print("\n>>> B1 CNN \n")
history_dict["b1"]["cnn"] = {}
results_dict["b1"]["cnn"] = {}
cnn_history_obj, cnn_results_obj = b1.cnn(b1_dir)
history_dict["b1"]["cnn"] = cnn_history_obj
results_dict["b1"]["cnn"] = cnn_results_obj
print("\n>>> B1 MobileNet \n")
history_dict["b1"]["mobilenet"] = {}
results_dict["b1"]["mobilenet"] = {}
mobilenet_history_obj, mobilenet_results_obj = b1.mobilenet(b1_dir)
history_dict["b1"]["mobilenet"] = mobilenet_history_obj
results_dict["b1"]["mobilenet"] = mobilenet_results_obj
print("\n>>> B1 MobileNetV2.\n")
history_dict["b1"]["mobilenetv2"] = {}
results_dict["b1"]["mobilenetv2"] = {}
mobilenetv2_history, mobilenetv2_results = b1.mobilenetv2(b1_dir)
history_dict["b1"]["mobilenetv2"] = mobilenetv2_history
results_dict["b1"]["mobilenetv2"] = mobilenetv2_results
print("\n>>> B1 NASNetMobile \n")
history_dict["b1"]["nasnetmobile"] = {}
results_dict["b1"]["nasnetmobile"] = {}
nasnetmobile_history, nasnetmobile_results = b1.nasnetmobile(b1_dir)
history_dict["b1"]["nasnetmobile"] = nasnetmobile_history
results_dict["b1"]["nasnetmobile"] = nasnetmobile_results

results_table.add_rows(
    [
        [
            "B1",
            "Training Acc",
            max(history_dict["b1"]["cnn"]["accuracy"]),
            max(history_dict["b1"]["mobilenet"]["accuracy"]),
            max(history_dict["b1"]["mobilenetv2"]["accuracy"]),
            max(history_dict["b1"]["nasnetmobile"]["accuracy"]),
        ],
        [
            "B1",
            "Validation Acc",
            max(history_dict["b1"]["cnn"]["val_accuracy"]),
            max(history_dict["b1"]["mobilenet"]["val_accuracy"]),
            max(history_dict["b1"]["mobilenetv2"]["val_accuracy"]),
            max(history_dict["b1"]["nasnetmobile"]["val_accuracy"]),
        ],
        [
            "B1",
            "Testing Acc",
            results_dict["b1"]["cnn"]["accuracy"],
            results_dict["b1"]["mobilenet"]["accuracy"],
            results_dict["b1"]["mobilenetv2"]["accuracy"],
            results_dict["b1"]["nasnetmobile"]["accuracy"],
        ],
    ]
)
print("\n>>> B1 Analysis \n")
all_b1_histories = [cnn_history_obj, mobilenet_history_obj, mobilenetv2_history, nasnetmobile_history]
all_b1_results = [cnn_results_obj, mobilenet_results_obj, mobilenetv2_results, nasnetmobile_results]
b1.analyse(b1_dir, all_b1_histories, all_b1_results)
print("\n>>> Cleaning Up B1 Task.\n")
cleanup(b1_dir)
print("Clean up memory/GPU etc...")  # TODO: Some code to free memory if necessary.
print('\n\n\n>>> LATEST RESULTS: \n\n\n')
print(results_table)



# ======================================================================================================================
# Task B2
print("\n~~~~~ Task B2 ~~~~~\n")
history_dict["b2"] = {}
results_dict["b2"] = {}
# Instantiate B2 task class
b2 = b2.B2()
# Data preprocessing for task B2
print("\n> Generating B2 Datasets\n")
b2.prepare(b2_dir, cartoon_set_img_dir, cartoon_set_df)
print("\n>>> B2 CNN \n")
history_dict["b2"]["cnn"] = {}
results_dict["b2"]["cnn"] = {}
cnn_history_obj, cnn_results_obj = b2.cnn(b2_dir)
history_dict["b2"]["cnn"] = cnn_history_obj
results_dict["b2"]["cnn"] = cnn_results_obj
print("\n>>> B2 MobileNet \n")
history_dict["b2"]["mobilenet"] = {}
results_dict["b2"]["mobilenet"] = {}
mobilenet_history_obj, mobilenet_results_obj = b2.mobilenet(b2_dir)
history_dict["b2"]["mobilenet"] = mobilenet_history_obj
results_dict["b2"]["mobilenet"] = mobilenet_results_obj
print("\n>>> B2 MobileNetV2.\n")
history_dict["b2"]["mobilenetv2"] = {}
results_dict["b2"]["mobilenetv2"] = {}
mobilenetv2_history, mobilenetv2_results = b2.mobilenetv2(b2_dir)
history_dict["b2"]["mobilenetv2"] = mobilenetv2_history
results_dict["b2"]["mobilenetv2"] = mobilenetv2_results
print("\n>>> B2 NASNetMobile \n")
history_dict["b2"]["nasnetmobile"] = {}
results_dict["b2"]["nasnetmobile"] = {}
nasnetmobile_history, nasnetmobile_results = b2.nasnetmobile(b2_dir)
history_dict["b2"]["nasnetmobile"] = nasnetmobile_history
results_dict["b2"]["nasnetmobile"] = nasnetmobile_results

results_table.add_rows(
    [
        [
            "B2",
            "Training Acc",
            max(history_dict["b2"]["cnn"]["accuracy"]),
            max(history_dict["b2"]["mobilenet"]["accuracy"]),
            max(history_dict["b2"]["mobilenetv2"]["accuracy"]),
            max(history_dict["b2"]["nasnetmobile"]["accuracy"]),
        ],
        [
            "B2",
            "Validation Acc",
            max(history_dict["b2"]["cnn"]["val_accuracy"]),
            max(history_dict["b2"]["mobilenet"]["val_accuracy"]),
            max(history_dict["b2"]["mobilenetv2"]["val_accuracy"]),
            max(history_dict["b2"]["nasnetmobile"]["val_accuracy"]),
        ],
        [
            "B2",
            "Testing Acc",
            results_dict["b2"]["cnn"]["accuracy"],
            results_dict["b2"]["mobilenet"]["accuracy"],
            results_dict["b2"]["mobilenetv2"]["accuracy"],
            results_dict["b2"]["nasnetmobile"]["accuracy"],
        ],
    ]
)
print("\n>>> B2 Analysis \n")
all_b2_histories = [cnn_history_obj, mobilenet_history_obj, mobilenetv2_history, nasnetmobile_history]
all_b2_results = [cnn_results_obj, mobilenet_results_obj, mobilenetv2_results, nasnetmobile_results]
b2.analyse(b2_dir, all_b2_histories, all_b2_results)
print("\n>>> Cleaning Up B2 Task.\n")
cleanup(b2_dir)
print("Clean up memory/GPU etc...")  # TODO: Some code to free memory if necessary.


# ======================================================================================================================
# TODO: Print out your results with following format:
print("\n~~~~~ Final Results ~~~~~\n")
print(results_table)
