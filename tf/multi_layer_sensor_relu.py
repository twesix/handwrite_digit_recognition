import tensorflow as tf
from mnist_data.numpy import train_get_batch, test_get_batch

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
x = tf.reshape(x, [-1, 784])
t = tf.placeholder(tf.float32, [None, 10])

l1 = 200
l2 = 100
l3 = 60
l4 = 30
l5 = 10

w1 = tf.Variable(tf.truncated_normal([28 * 28, l1], stddev=0.1))
b1 = tf.Variable(tf.ones([l1]) / 10)

w2 = tf.Variable(tf.truncated_normal([l1, l2], stddev=0.1))
b2 = tf.Variable(tf.ones([l2]) / 10)

w3 = tf.Variable(tf.truncated_normal([l2, l3], stddev=0.1))
b3 = tf.Variable(tf.ones([l3]) / 10)

w4 = tf.Variable(tf.truncated_normal([l3, l4], stddev=0.1))
b4 = tf.Variable(tf.ones([l4]) / 10)

w5 = tf.Variable(tf.truncated_normal([l4, l5], stddev=0.1))
b5 = tf.Variable(tf.ones([l5]) / 10)

y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
logits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(logits)

init_op = tf.global_variables_initializer()

cross_entropy = - tf.reduce_sum(t * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
train_step = optimizer.minimize(cross_entropy)

batch_size = 100
with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(10000):
        _ = _ + 1
        _Xs, _Ys = train_get_batch(batch_size)
        sess.run(train_step, feed_dict={x: _Xs.reshape(-1, 784), t: _Ys})

        if _ % 10 == 0:
            _Xs, _Ys = test_get_batch(10000)
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: _Xs.reshape(-1, 784), t: _Ys})
            print('round: %s, acc: %s' % (_, acc))
        pass

