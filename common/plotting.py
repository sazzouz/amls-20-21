import os
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from tensorflow.keras.utils import plot_model

plt.style.use("seaborn")

def plot_single_accuracy_loss(dir, history, task_name, model_name='Model'):
    # Plot accuracies
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    # summarize history for accuracy
    ax.plot(history.history['accuracy'], 'b-', label="acc")
    ax.plot(history.history['val_accuracy'], 'b--', label="acc")
    ax.plot(history.history['loss'], 'g-', label="acc")
    ax.plot(history.history['val_loss'], 'g--', label="acc")
    ax.set_title('{0} Train / Val Accuracy & Loss'.format(model_name))
    ax.set_ylabel('Accuracy / Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['{0}_acc'.format(model_name), '{0}_val_acc'.format(model_name), '{0}_loss'.format(model_name), '{0}_val_loss'.format(model_name)], loc='center right')
    fig.savefig(os.path.join(dir, 'results', '{0}_{1}_accuracy_loss'.format(task_name, model_name)), bbox_inches='tight')

def plot_all_accuracy(dir, all_histories, task_name):
    # Plot accuracies
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    # summarize history for accuracy
    ax.plot(all_histories[0]['accuracy'], 'b-')
    ax.plot(all_histories[0]['val_accuracy'], 'b--')
    ax.plot(all_histories[1]['accuracy'], 'g-')
    ax.plot(all_histories[1]['val_accuracy'], 'g--')
    ax.plot(all_histories[2]['accuracy'], 'm-')
    ax.plot(all_histories[2]['val_accuracy'], 'm--')
    ax.plot(all_histories[3]['accuracy'], 'y-')
    ax.plot(all_histories[3]['val_accuracy'], 'y--')
    ax.set_title('Model Train / Val Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['CNN_acc', 'CNN_val_acc', 'MobileNet_acc', 'MobileNet_val_acc', 'MobileNetV2_acc', 'MobileNetV2_val_acc', 'NASNetMobile_acc', 'NASNetMobile_val_acc'], loc='lower right')
    fig.savefig(os.path.join(dir, 'results', '{0}_all_accuracy'.format(task_name)), bbox_inches='tight')

def plot_all_loss(dir, all_histories, task_name):
    # Plot losses
    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    # summarize history for losses
    ax.plot(all_histories[0]['loss'], 'b-')
    ax.plot(all_histories[0]['val_loss'], 'b--')
    ax.plot(all_histories[1]['loss'], 'g-')
    ax.plot(all_histories[1]['val_loss'], 'g--')
    ax.plot(all_histories[2]['loss'], 'm-')
    ax.plot(all_histories[2]['val_loss'], 'm--')
    ax.plot(all_histories[3]['loss'], 'y-')
    ax.plot(all_histories[3]['val_loss'], 'y--')
    ax.set_title('Model Train / Val loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['CNN_loss', 'CNN_val_loss', 'MobileNet_loss', 'MobileNet_val_loss', 'MobileNetV2_loss', 'MobileNetV2_val_loss', 'NASNetMobile_loss', 'NASNetMobile_val_loss'], loc='upper right')
    fig.savefig(os.path.join(dir, 'results', '{0}_all_loss'.format(task_name)), bbox_inches='tight')

