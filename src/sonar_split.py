from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import  LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

#seed
seed = 526
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/sonar.csv", header=None)
#print(df.head(5))

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=150, batch_size=5)

print("\nAccuracy : {}".format(model.evaluate(X_test, Y_test)[1]))