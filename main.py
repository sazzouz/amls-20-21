# Import dependancies for main file.
import click # Library to compose CLI functionality.

# Import class defintions for each task.
from A1 import a1
from A2 import a2
from B1 import b1
from B2 import b2

# ======================================================================================================================
# Task A1
print('\n~~~~~ Task A1 ~~~~~\n')
# Data preprocessing
a1_data_train, a1_data_val, a1_data_test = a1.prepare() # TODO: Data preparation step for A1.
## model_A1 = A1(args...) # TODO: Build model object.
model_A1 = a1.A1()
# Train model based on the training set (you should fine-tune your model based on validation set.)
## acc_A1_train = model_A1.train(args...)  # TODO: Train model based on the train set.
a1_train_params = {}
acc_A1_train = model_A1.train(a1_train_params)
## acc_A1_test = model_A1.test(args...)  # TODO: Test model based on the test set.
a1_test_params = {}
acc_A1_test = model_A1.test(a1_test_params)
print('Clean up memory/GPU etc...')  # TODO: Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
print('\n~~~~~ Task A2 ~~~~~\n')
a2_data_train, a2_data_val, a2_data_test = a2.prepare() # TODO: Data preparation step for A2.
## model_A1 = A1(args...) # TODO: Build model object.
model_A2 = a2.A2()
# Train model based on the training set (you should fine-tune your model based on validation set.)
## acc_A1_train = model_A1.train(args...)  # TODO: Train model based on the train set.
a2_train_params = {}
acc_A2_train = model_A2.train(a2_train_params)
## acc_A1_test = model_A1.test(args...)  # TODO: Test model based on the test set.
a2_test_params = {}
acc_A2_test = model_A2.test(a2_test_params)
print('Clean up memory/GPU etc...')  # TODO: Some code to free memory if necessary.


# ======================================================================================================================
# Task B1
print('\n~~~~~ Task B1 ~~~~~\n')
b1_data_train, b1_data_val, b1_data_test = b1.prepare() # TODO: Data preparation step A3.
## model_A1 = A1(args...) # TODO: Build model object.
model_B1 = b1.B1()
# Train model based on the training set (you should fine-tune your model based on validation set.)
## acc_A1_train = model_A1.train(args...)  # TODO: Train model based on the train set.
b1_train_params = {}
acc_B1_train = model_B1.train(b1_train_params)
## acc_A1_test = model_A1.test(args...)  # TODO: Test model based on the test set.
b1_test_params = {}
acc_B1_test = model_B1.test(b1_test_params)
print('Clean up memory/GPU etc...')  # TODO: Some code to free memory if necessary.


# ======================================================================================================================
# Task B2
print('\n~~~~~ Task B2 ~~~~~\n')
b2_data_train, b2_data_val, b2_data_test = b2.prepare() # TODO: Data preparation step A4.
## model_A1 = A1(args...) # TODO: Build model object.
model_B2 = b2.B2()
# Train model based on the training set (you should fine-tune your model based on validation set.)
## acc_A1_train = model_A1.train(args...)  # TODO: Train model based on the train set.
b2_train_params = {}
acc_B2_train = model_B2.train(b2_train_params)
## acc_A1_test = model_A1.test(args...)  # TODO: Test model based on the test set.
b2_test_params = {}
acc_B2_test = model_B2.test(b2_test_params)
print('Clean up memory/GPU etc...')  # TODO: Some code to free memory if necessary.


# ======================================================================================================================
# TODO: Print out your results with following format:
print('\n~~~~~ Results ~~~~~\n')
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};\n'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
print('\n(Results summaries available in ./results)\n')
