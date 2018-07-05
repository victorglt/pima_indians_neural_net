import pimaindians_dataset as pima
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


def display_accuracy_graph(training_acc, validation_acc):
    plt.clf()

    epochs = range(1, len(training_acc) + 1)

    plt.plot(epochs, training_acc, 'bo', label='Training acc')
    plt.plot(epochs, validation_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def display_loss_graph(training_loss, validation_loss):
    plt.clf()

    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, 'bo', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def show_metrics(history):
    history_dict = history.history
    history_dict.keys()

    display_loss_graph(history.history['loss'], history.history['val_loss'])
    display_accuracy_graph(history.history['acc'], history.history['val_acc'])


model = Sequential()

(x_train, y_train), (x_test, y_test) = pima.load_data(test_percentage=0.1)

model = Sequential()

model.add(Dense(32, activation='relu', input_dim=8))

# Add fully connected layer with a ReLU activation function
model.add(Dense(32, activation='relu'))

# Add fully connected layer with a sigmoid activation function
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_test, y_test))

model.save('diabetes_model.h5')

show_metrics(history)

