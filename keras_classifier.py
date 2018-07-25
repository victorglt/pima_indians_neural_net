import pimaindians_dataset as pima
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold


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

    model.add(Dense(64, activation='relu', input_dim=8))

    # Add fully connected layer with a ReLU activation function
    model.add(Dense(64, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def k_fold_train(model, x_train, y_train, show_metric_graphs=False):
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

#test_percentage set to 0 as to load the full data. Splitting will be done by K-Fold
(x_train, y_train), (x_test, y_test) = pima.load_data(test_percentage=0)

model = build_model()
(mean_acc, std_deviation) = k_fold_train(model, x_train, y_train)

print "Your model has acc of: " + str(mean_acc * 100) + "% with a standard deviation of: " + str(std_deviation * 100) + "%"

model.save("diabetes_model.h5")