import argparse
import pimaindians_dataset as pima
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model

DEFAULT_MODEL_NAME = "diabetes_model.h5"

def display_accuracy_graph(training_acc, validation_acc):
    epochs = range(1, len(training_acc) + 1)

    plt.plot(epochs, training_acc, 'bo', label='Training acc')
    plt.plot(epochs, validation_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def display_loss_graph(training_loss, validation_loss):
    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, 'bo', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def show_metrics(history):
    print(history.history)

    plt.figure(1)
    plt.subplot(211)
    display_loss_graph(history.history['loss'], history.history['val_loss'])

    plt.subplot(212)
    display_accuracy_graph(history.history['acc'], history.history['val_acc'])

    plt.show()


def build_model():
    model = Sequential()

    model.add(Dense(16, activation='relu', input_dim=8))

    # Add fully connected layer with a ReLU activation function
    model.add(Dense(16, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def k_fold_train(model, x_train, y_train, show_metric_graphs=True):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=128)

    k_fold_accuracies = []

    for k_train, k_test in kfold.split(x_train,y_train):
        history = model.fit(
            x_train[k_train],
            y_train[k_train],
            epochs=10,
            batch_size=512,
            validation_data=(x_train[k_test], y_train[k_test]))

        #Get the acc of the last epoch
        val_acc = history.history['val_acc'][-1]
        k_fold_accuracies.append(val_acc)

        if show_metric_graphs:
            show_metrics(history)
    return (np.mean(k_fold_accuracies), np.std(k_fold_accuracies))


def infer(parameters, model_name):
    if model_name == None:
        model_name = DEFAULT_MODEL_NAME
    model = load_model(model_name)
    return model.predict(parameters)

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Trains the network", action="store_true")
parser.add_argument("--graphs", help="Display metrics graphs at each K-Fold iteration", action="store_true")
parser.add_argument("--predict", help="Predicts from saved model", action="store_true")
parser.add_argument("--model", help="Name of the mode to do prediction", metavar="M", nargs=1, dest="model_name", action="store")

args = parser.parse_args()

if args.train:
    (x_train, y_train), (x_test, y_test) = pima.load_data(test_percentage=0)
    (mean_acc, std_deviation) = k_fold_train(build_model(), x_train, y_train, args.graphs)     
    print("Your model has acc of: " + str(mean_acc * 100) + "% with a standard deviation of: " + str(std_deviation * 100) + "%")
if args.predict:
    console_input = input("Prediction Input, space separated:")
    prediction_input = np.array([console_input.split(" ")])    
    prediction_result = infer(prediction_input, args.model_name)
    print("The prediction for: " + str(prediction_input) + " is: " + str(prediction_result))











