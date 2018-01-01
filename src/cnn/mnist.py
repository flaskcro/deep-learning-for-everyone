from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np

#Load MNIST Data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("Trainning Image Count : {}, Test Image Count : {}".
      format(X_train.shape[0], X_test.shape[0]))

#preprocessing / One-Hot Encoding
X_train = X_train.reshape(X_train.shape[0], 784).astype('float64') / 255
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float64') / 255

Y_train = np_utils.to_categorical(Y_train,10)
Y_test = np_utils.to_categorical(Y_test,10)

#Making Model
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch:02d}-{loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200,
                    verbose=0, callbacks=[early_stopping_callback, checkpointer])

import matplotlib.pyplot as plt

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_vacc = history.history['val_acc']
y_acc = history.history['acc']
x_len = np.arange(len(y_loss))

print("\nTrainning Accuracy : {:.2f}%, Test Accuracy : {:.2f}%".format(y_acc[len(y_loss)-1]*100.0, y_vacc[len(y_loss)-1]*100.0))
print("\nTrainning Loss : {:.2f}%, Test Loss : {:.2f}%".format(y_loss[len(y_loss)-1]*100.0, y_vloss[len(y_loss)-1]*100.0))

plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
plt.plot(x_len, y_vacc, marker='.', c='orange', label='Testset_accuracy')
plt.plot(x_len, y_acc, marker='.', c='green', label='Trainset_accuracy')

plt.legend(loc='center right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()