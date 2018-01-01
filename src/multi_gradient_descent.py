import tensorflow as tf

data = [[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]
z_data = [i[2] for i in data]

learning_rate = 0.1
x_slope = tf.Variable(tf.random_uniform([1], 0, 10, dtype= tf.float64, seed=0))
y_slope = tf.Variable(tf.random_uniform([1], 0, 10, dtype= tf.float64, seed=0))
init_intercept =  tf.Variable(tf.random_uniform([1], 0, 100, dtype= tf.float64, seed=0))

y = x_slope * x_data + y_slope * y_data + init_intercept
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - z_data)))

gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(gradient_descent)
        if step % 100 == 0:
            print("Epoch : {}, RMSE = {}, Slope x = {}, , Slope y = {}, Intercept = {}".
                  format( step, sess.run(rmse), sess.run(x_slope), sess.run(y_slope), sess.run(init_intercept)))
    sess.close()