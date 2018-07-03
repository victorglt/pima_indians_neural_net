import pimaindians_dataset as pima
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers
from keras import metrics 
from keras import losses

model = Sequential()

(x_train, y_train), (x_test, y_test) = pima.load_data(test_percentage=0.1)

print "X Train Data Shape: " + str(x_train.shape)
print "Y Train Data Shape: " + str(y_train.shape)
print "X Test Data Shape: " + str(x_test.shape)
print "Y Test Data Shape: " + str(y_test.shape)

print x_train
print y_train 

model = Sequential()

model.add(Dense(16, activation='relu', input_dim=8))

# Add fully connected layer with a ReLU activation function
model.add(Dense(16, activation='relu'))

model.add(Dense(16, activation='relu'))

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


print model.predict(x_test)


history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

