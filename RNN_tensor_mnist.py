#!python3
"""
RNN_tensor_mnist.py
Network from pythonprogramming.net
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import configparser
import codeTime as cT

# suppress warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = configparser.ConfigParser()
config.read('config.ini')
DATA_FOLDER_PATH = [config.get('main', 'mlpath'), config.get('main', 'mnist')]
DATA_FOLDER_PATH = os.path.join(*DATA_FOLDER_PATH)
MODEL_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'models', 'rnn')
os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
mnist = input_data.read_data_sets(DATA_FOLDER_PATH, one_hot=True)
rnnmodelsave = "rnnmodel.ckpt"

hm_epochs = 20
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 512

# height x width (28 * 28) for mnistExp
x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
y = tf.placeholder(tf.float32)


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes], dtype=tf.float32), dtype=tf.float32),
             'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float32), dtype=tf.float32)}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights'] + layer['biases'])

    return output


@cT.exeTime
def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, 'loss:', epoch_loss)

        # Perform save
        save_path = saver.save(sess, os.path.join(MODEL_FOLDER_PATH, rnnmodelsave))
        print("Model saved in file: %s" % save_path)

        # Run
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:',
              accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                             y: mnist.test.labels}))


def test_load_neural_network():
    x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
    prediction = recurrent_neural_network(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(MODEL_FOLDER_PATH, rnnmodelsave))
        print("Model restored.")

        # Run
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                                          y: mnist.test.labels}))


def use_neural_network_list(imgdata_list):
    x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
    # Initialize nn-model
    prediction = recurrent_neural_network(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore nn-model from file
        saver.restore(sess, os.path.join(MODEL_FOLDER_PATH, rnnmodelsave))
        print("Model restored.")

        outputLabels = []
        for imgdata in imgdata_list:
            # Get label for image data
            # prediction = neural_network_model(imgdata)
            imgdata = imgdata.reshape((-1, n_chunks, chunk_size))
            mxp = tf.argmax(prediction, 1)
            outputLabels.append(mxp.eval(feed_dict={x: imgdata}))

        return outputLabels


if __name__ == "__main__":
    # Accuracy: 0.9887 with 20 epochs
    train_neural_network(x)
    #
    # # Accuracy: 0.9887
    # test_load_neural_network()
    pass
