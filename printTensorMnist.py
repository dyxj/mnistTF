#!python3
"""
printTensorMnist.py
"""
from tensorflow.examples.tutorials.mnist import input_data
import configparser
import os
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
DATA_FOLDER_PATH = [config.get('main', 'mlpath'), config.get('main', 'mnist')]
DATA_FOLDER_PATH = os.path.join(*DATA_FOLDER_PATH)
mnist = input_data.read_data_sets(DATA_FOLDER_PATH, one_hot=True)

# Get label of mnist data set 0-9999
imgindex = 0
label = mnist.test.labels[imgindex]
print(label)
label = [i for i, l in enumerate(label) if l != 0]
print("image label: {}".format(label))
# for i, l in enumerate(label):
#     if l != 0:
#         print(i)
testimg = mnist.test.images[imgindex].reshape(1, 784)
print("output label: {}".format(label))

# Show image
testimg = testimg.reshape(28, 28)
plt.figure("ImageIndex: {}  |  Label: {}  |  Predicted Label: {}"
           .format(imgindex, label, label))
plt.imshow(testimg, cmap='gray')
plt.show()
