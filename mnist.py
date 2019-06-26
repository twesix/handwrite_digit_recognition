import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

x = tf.reshape(x, [-1, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init_op = tf.global_variables_initializer()

y = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = - tf.reduce_sum(t * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
train_step = optimizer.minimize(cross_entropy)

with tf.Session as sess:
    sess.run(init_op)
    for _ in range(1000):
        pass
