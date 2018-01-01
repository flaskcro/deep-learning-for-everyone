import pandas as pd
import numpy as np
import tensorflow as tf;
from keras.models import Sequential
from keras.layers import Dense

#generate seed
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#data load
data = np.loadtxt("../dataset/pima-indians-diabetes.data", delimiter=',')

X = data[:,0:8]
Y = data[:,8]

#Model define
model = Sequential()
model.add(Dense(12, input_dim= 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Model Compile
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

#Model execute
model.fit(X, Y, epochs= 300, batch_size= 10)

print("\n Accuray : {}".format(model.evaluate(X,Y)[1]))