from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow as tf

#generate seed
seed = 20171230
np.random.seed(seed)
tf.set_random_seed(seed)

ds = np.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

X = ds[:,0:17]
Y = ds[:,17]

#Making Model
model = Sequential()
model.add(Dense(30,input_dim = 17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#Model Trainning
#model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=3000, batch_size= 100)

#print result
print("\n Accuracy : %.4f" % (model.evaluate(X, Y))[1])