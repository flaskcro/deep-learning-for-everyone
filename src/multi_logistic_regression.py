import tensorflow as tf
import numpy as np

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

x_data = np.array([[2,3], [4,3], [6,4], [8,6], [10,7], [12,8], [14,9]])
y_data = np.array([0,0,0,1,1,1,1]).reshape(7,1)

X = tf.placeholder(tf.float64, shape=[None,2])
Y = tf.placeholder(tf.float64, shape=[None,1])

a = tf.Variable(tf.random_uniform([2,1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

#sigmoid
y = tf.sigmoid(tf.matmul(X,a) + b)

#loss
loss = -tf.reduce_mean( Y * tf.log(y) + (1 - Y) * tf.log(1-y))

learning_rate = 0.1
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(3001):
		a_, b_, loss_, _ = sess.run([a,b, loss, gradient_descent], feed_dict={X: x_data, Y: y_data})
		if (i + 1) % 300 == 0:
			print("Step : {}, a1 = {}, a2 = {}, b = {}, loss = {}".format(i+1, a_[0], a_[1], b_, loss_))

	new_x = np.array([7, 6]).reshape(1, 2)
	new_y = sess.run(y, feed_dict={X: new_x})

	print("Study hours {}, Private Lesson count {}".format(new_x[:,0], new_x[:,1]))
	print("Pass rate {}".format(new_y * 100.0))

	sess.close()

