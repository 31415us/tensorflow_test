
import tensorflow as tf
from tensorflow.contrib import rnn

from dataprep import CHARCOUNT, SentenceBatcher, print_inference


embed_dim = 1
max_len = 128
n_hidden = 10
n_classes = 5
n_char = CHARCOUNT

x = tf.placeholder(tf.int32, [None, max_len, 1])
y = tf.placeholder(tf.float32, [None, max_len, n_classes])
#seqlen = tf.placeholder(tf.int32, [None])

weights = {
        'out' : tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}

biases = {
        'out' : tf.Variable(tf.random_normal([n_classes]))
}

embedding = tf.Variable(tf.random_normal([CHARCOUNT, embed_dim]))

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def blstm(x, weights, biases):

    ins = tf.nn.embedding_lookup(embedding, x)
    ins = tf.reshape(ins, [-1, max_len, embed_dim])

    fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, ins, dtype=tf.float32, sequence_length=length(x))
    outputs = tf.concat(outputs, 2)

    W = weights['out']
    W = tf.reshape(W, [1, 2*n_hidden, n_classes])
    n = tf.shape(outputs)[0]
    W = tf.tile(W, [n, 1, 1])
    res = tf.matmul(outputs, W)

    b = biases['out']
    b = tf.reshape(b, [1, 1, n_classes])
    b = tf.tile(b, [n, max_len, 1])

    res = tf.add(res, b)

    res = tf.nn.log_softmax(res, dim=-1)

    mask = tf.sequence_mask(length(x), maxlen=max_len, dtype=tf.bool)
    mask = tf.reshape(mask, [-1, max_len, 1])
    mask = tf.cast(mask, dtype=tf.float32)

    res = tf.multiply(res, mask)

    return res

def simple_cost(log_likelihoods, y):
    res = tf.multiply(log_likelihoods, y)
    res = tf.reduce_sum(res, axis=-1)
    res = tf.reduce_sum(res, axis=-1)
    res = tf.reduce_sum(res, axis=-1)

    return - res

def simple_inference(pred):
    label_num = tf.argmax(pred, axis=-1)

    return label_num

def accuracy(pred, y):
    pred_labels = tf.argmax(pred, axis=-1)
    true_labels = tf.argmax(y, axis=-1)

    diff = tf.sign(tf.abs(pred_labels - true_labels))

    num_diff = tf.reduce_sum(diff, axis=-1)

    acc = tf.divide(tf.cast(num_diff, tf.float32), tf.cast(length(x), tf.float32))

    acc = 1 - tf.reduce_mean(acc)

    return acc


pred = blstm(x, weights, biases)

cost = simple_cost(pred, y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

inf = simple_inference(pred)

accur = accuracy(pred, y)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    dataset = SentenceBatcher('input.txt', max_len)

    for i in range(1000):
        xin, yin, _, _ = dataset.next_batch(10)

        d = {x: xin, y: yin}

        sess.run(optimizer, feed_dict=d)

        if i % 10 == 0:
            loss = sess.run(cost, feed_dict=d)
            print('iter ' + str(i) + ' cost ' + str(loss))

    print('==== FINISH TRAINING ====')

    xin, yin, ss, ts = dataset.next_batch(10)
    d = {x: xin, y: yin}

    labels = sess.run(inf, feed_dict=d)

    print_inference(ss, ts, labels)

    acc = sess.run(accur, feed_dict=d)

    print(acc)

