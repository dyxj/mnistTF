#!python3
"""
RNN_tensor_mnist.py
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import configparser

# suppress warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = configparser.ConfigParser()
config.read('config.ini')
DATA_FOLDER_PATH = [config.get('main', 'mlpath'), config.get('main', 'mnist')]
DATA_FOLDER_PATH = os.path.join(*DATA_FOLDER_PATH)
MODEL_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'models', 'mp')
os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
mnist = input_data.read_data_sets(DATA_FOLDER_PATH, one_hot=True)
nnmodelsave = "model.ckpt"

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100
hm_epochs = 20

# height x width (28 * 28) for mnistExp
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    # (input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
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
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, 'loss:', epoch_loss)

        save_path = saver.save(sess, os.path.join(MODEL_FOLDER_PATH, nnmodelsave))
        print("Model saved in file: %s" % save_path)

        # Run
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


def test_load_neural_network(x):
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(MODEL_FOLDER_PATH, nnmodelsave))
        print("Model restored.")

        # Run
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


def use_neural_network(imgdata):
    # Initialize nn-model
    prediction = neural_network_model(imgdata)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore nn-model from file
        saver.restore(sess, os.path.join(MODEL_FOLDER_PATH, nnmodelsave))
        print("Model restored.")

        # Get label for image data
        mxp = tf.argmax(prediction, 1)
        return mxp.eval(feed_dict={x: imgdata})


def use_neural_network_list(imgdata_list):
    x1 = tf.placeholder('float', [None, 784])
    # Initialize nn-model
    prediction = neural_network_model(x1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore nn-model from file
        saver.restore(sess, os.path.join(MODEL_FOLDER_PATH, nnmodelsave))
        print("Model restored.")

        outputLabels = []
        for imgdata in imgdata_list:
            # Get label for image data
            # prediction = neural_network_model(imgdata)
            mxp = tf.argmax(prediction, 1)
            outputLabels.append(mxp.eval(feed_dict={x1: imgdata}))

        return outputLabels


if __name__ == "__main__":
    # Accuracy: 0.9673 with 50 epochs
    train_neural_network(x)

    # # Accuracy: 0.9673
    # test_load_neural_network(x)

    # # Get label of mnist data set 0-9999
    # imgindex = 0
    # label = mnist.test.labels[imgindex]
    # print(label)
    # label = [i for i, l in enumerate(label) if l != 0]
    # print("image label: {}".format(label))
    # # for i, l in enumerate(label):
    # #     if l != 0:
    # #         print(i)
    # testimg = mnist.test.images[imgindex].reshape(1, 784)
    # outputlabel = use_neural_network(testimg)
    # print("output label: {}".format(outputlabel))
    #
    # # Show image
    # testimg = testimg.reshape(28, 28)
    # plt.figure("ImageIndex: {}  |  Label: {}  |  Predicted Label: {}"
    #            .format(imgindex, label, outputlabel))
    # plt.imshow(testimg, cmap='gray')
    # plt.show()

    pass