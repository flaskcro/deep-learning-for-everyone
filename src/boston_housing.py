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

df = pd.read_csv("../dataset/housing.csv", delim_whitespace=True, header=None)

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

early_stopping = EarlyStopping(monitor='loss', patience=50)
model.compile(loss="mean_squared_error", optimizer='adam')
history = model.fit(X_train, Y_train, epochs=3000, batch_size=10, callbacks=[early_stopping])

Y_pred = model.predict(X_test).flatten()

for i in range(10):
	label = Y_test[i]
	prediction = Y_pred[i]
	print("Actual Price : {}, Prediction Price : {}".format(label, prediction))

