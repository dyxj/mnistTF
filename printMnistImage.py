#!python3
"""
Guide for importing mnist from original source
- Print an element from test image

data can be obtained from : http://yann.lecun.com/exdb/mnist/
Image file:
images are 28 * 28.
Hence every 784 bytes after the first 16 bytes represent an image
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import traceback
import gzip
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
DATA_FOLDER_PATH = [config.get('main', 'mlpath'), config.get('main', 'mnist')]
DATA_FOLDER_PATH = os.path.join(*DATA_FOLDER_PATH)
print(DATA_FOLDER_PATH)

DATA_TEST_IMG_PATH = os.path.join(DATA_FOLDER_PATH, r"t10k-images-idx3-ubyte.gz")
DATA_TEST_LABEL_PATH = os.path.join(DATA_FOLDER_PATH, r"t10k-labels-idx1-ubyte.gz")
DATA_TRAIN_IMG_PATH = os.path.join(DATA_FOLDER_PATH, r"train-images-idx3-ubyte.gz")
DATA_TRAIN_LABEL_PATH = os.path.join(DATA_FOLDER_PATH, r"train-labels-idx1-ubyte.gz")


def load_mnist_label(path):
    with gzip.open(path, 'rb') as file:
        # Get data from bytes of 4
        # magic number, label count
        magic, lcount = struct.unpack(">II", file.read(8))
        print(str(magic) + " " + str(lcount))
        # Compare magic number
        if magic != 2049:
            raise ValueError("Magic number mismatch, expected 2049 for label data, value obtained {}".format(magic))

        data = np.frombuffer(file.read(), np.uint8)

    return data


def load_mnist_img(path):
    with gzip.open(path, 'rb') as file:
        # Get data from bytes of 4
        # magic number, image count, row size, column size
        magic, imgcount, rows, cols = struct.unpack(">IIII", file.read(16))
        print(str(magic) + " " + str(imgcount) + " " + str(rows) + " " + str(cols))
        # Compare magic number
        if magic != 2051:
            raise ValueError("Magic number mismatch, expected 2051 for image data, value obtained {}".format(magic))

        data = np.frombuffer(file.read(), np.uint8)

    data = data.reshape(-1, 28, 28)
    return data


if __name__ == "__main__":
    try:
        # Load Test labels and Test images
        testLabels = load_mnist_label(DATA_TEST_LABEL_PATH)
        testImages = load_mnist_img(DATA_TEST_IMG_PATH)

        # 0 - 9999
        testNo = 9999;

        # Print label
        print(testLabels[testNo])

        # Draw first and second image
        plt.figure("Image : " + str(testNo) + " | Label: " + str(testLabels[testNo]))
        plt.imshow(testImages[testNo], cmap='gray')
        plt.show()

    except Exception as e:
        emsg = traceback.format_exc()
        print(e)
        print(emsg)
