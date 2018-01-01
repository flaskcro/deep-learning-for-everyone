import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

seed = 526
np.random.seed(seed)
tf.set_random_seed(seed)

MODEL_DIR = "./model"
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

modelpath = "./model/wine-{epoch:02d}-{loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='loss', verbose=1, save_best_only=True)

df = pd.read_csv("../dataset/wine.csv", header=None)
dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

early_stopping = EarlyStopping(monitor="loss", patience=100)
history = model.fit(X_train,Y_train, epochs=3000, batch_size=500, callbacks=[checkpointer, early_stopping], verbose=0)

print("\nAccuracy : {}".format(model.evaluate(X_test,Y_test)[1]))

import matplotlib.pyplot as plt
y_loss = history.history['loss']
y_acc = history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_loss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)
plt.show()