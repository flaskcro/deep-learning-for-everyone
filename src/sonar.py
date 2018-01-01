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

from sklearn.model_selection import StratifiedKFold

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []
for train, test in skf.split(X, Y):
	model = Sequential()
	model.add(Dense(24, input_dim=60, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
	model.fit(X[train], Y[train], epochs=100, batch_size=5)
	k_accuracy = (model.evaluate(X[test], Y[test])[1])
	accuracy.append(k_accuracy)

	# model save
	model.save("sonar_model_10.h5")

print("\n{} fold accuracy : {}".format(n_fold, accuracy))

