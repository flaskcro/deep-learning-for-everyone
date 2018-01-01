import tensorflow as tf
import numpy as np

data = [ [2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1] ]

x_data = [x[0] for x in data]
y_data = [y[1] for y in data]

a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

#sigmoid
y = 1 / ( 1 + np.e ** (a* x_data + b))

#loss
loss = -tf.reduce_mean( np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1-y))

learning_rate = 0.5
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(60001):
		sess.run(gradient_descent)
		if i % 6000 == 0:
			print("Epoch : {}, loss : {}, slope : {}, incercept : {}". format(i, sess.run(loss), sess.run(a), sess.run(b)))

	sess.close()
