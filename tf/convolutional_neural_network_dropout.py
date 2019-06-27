import tensorflow as tf
from mnist_data.numpy import train_get_batch, test_get_batch
var = tf.Variable
placeholder = tf.placeholder

x = placeholder(tf.float32, [None, 28, 28, 1])
label = placeholder(tf.float32, [None, 10])
pkeep = placeholder(tf.float32)

L1 = 4
L2 = 8
L3 = 12
L4 = 200
L5 = 10

w1 = var(tf.truncated_normal([5, 5, 1, L1], stddev=0.1))
b1 = var(tf.ones([L1]) / 10)

w2 = var(tf.truncated_normal([5, 5, L1, L2], stddev=0.1))
b2 = var(tf.ones([L2])/10)

w3 = tf.Variable(tf.truncated_normal([4, 4, L2, L3], stddev=0.1))
b3 = tf.Variable(tf.ones([L3])/10)

w4 = tf.Variable(tf.truncated_normal([7 * 7 * L3, L4], stddev=0.1))
b4 = tf.Variable(tf.ones([L4])/10)

w5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([L5])/10)


y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)

yy = tf.reshape(y3, shape=[-1, 7 * 7 * L3])
y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
y4d = tf.nn.dropout(y4, pkeep)

logits = tf.matmul(y4d, w5) + b5
y = tf.nn.softmax(logits)

cross_entropy = - tf.reduce_sum(label * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

batch_size = 50
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        i = i + 1
        _Xs, _Ys = train_get_batch(batch_size)
        sess.run(train_step, feed_dict={x: _Xs, label: _Ys, pkeep: 0.6})

        if i % 100 == 0:
            _Xs, _Ys = test_get_batch(10000)
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: _Xs, label: _Ys, pkeep: 1})
            print('round: %s, acc: %s' % (i, acc))
