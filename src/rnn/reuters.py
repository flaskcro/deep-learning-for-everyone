from keras.datasets import reuters
import numpy as np

(Xtrain, Ytrain), (Xtest, Ytest) = reuters.load_data(num_words=1000, test_split=0.2)

from keras.preprocessing import sequence
from keras.utils import np_utils

x_train = sequence.pad_sequences(Xtrain, maxlen=100)
x_test = sequence.pad_sequences(Xtest, maxlen=100)

y_train = np_utils.to_categorical(Ytrain)
y_test = np_utils.to_categorical(Ytest)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

modelpath = MODEL_DIR + "reuters.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test),
                    callbacks=[early_stopping_callback, checkpointer])


print("\nTest Accuracy : %.4f" % (model.evaluate(x_test, y_test))[1])

import matplotlib.pyplot as plt
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_vacc = history.history['val_acc']
y_acc = history.history['acc']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='Test set Loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Train set Loss')
#plt.plot(x_len, y_vacc, marker='.', c='orange', label='Test set Accuracy')
#plt.plot(x_len, y_acc, marker='.', c='green', label='Train set Accuracy')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()