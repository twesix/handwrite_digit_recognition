import tensorflow as tf
from mnist_data.numpy import train_get_batch, test_get_batch

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

x = tf.reshape(x, [-1, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init_op = tf.global_variables_initializer()

y = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = - tf.reduce_sum(t * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
train_step = optimizer.minimize(cross_entropy)

batch_size = 100
with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(1000):
        _Xs, _Ys = train_get_batch(batch_size)
        sess.run(train_step, feed_dict={x: _Xs.reshape(-1, 784), t: _Ys})

        if _ % 100 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: _Xs.reshape(-1, 784), t: _Ys})
            print('acc: %s, loss: %s' % (acc, loss))
            # print(type(acc))
        pass

