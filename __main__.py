#! python3
"""
main for mnistExp : mnist experiment with tensor flow
Network from pythonprogramming.net
"""
import deepnet_mnist_wSave as deepNet
import RNN_tensor_mnist as rnnNet
import matplotlib.pyplot as plt
import os
import formatImage as fI
from PIL import Image as pImage


def main():
    # Test with hand(mspaint) drawn image
    folder = "hand"  # folder with test images
    image_path_list = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path_list.append(os.path.join(folder, filename))

    run_batch(image_path_list, mode="rnn", display="text")
    # run_batch(image_path_list, mode="mp", display="text")

    # Test with mnist data set
    # mnistIndex_list = list(range(10, 20))
    # runBatchMnist(mnistIndex_list, mode="deep", display="text")


def run_batch(image_path_list, mode="rnn", display="text"):
    imgdata_list = []
    for path in image_path_list:
        img = pImage.open(path).convert('L')
        imgdata = fI.format_img_mnist_data(img)
        imgdata_list.append(imgdata)

    # Use recursive neural network
    # or multilayer perceptron neural network
    if mode.lower() == "rnn":
        output_labels = rnnNet.use_neural_network_list(imgdata_list)
    else:
        output_labels = deepNet.use_neural_network_list(imgdata_list)

    # Print in console or display with matplotlib
    if display.lower() == "text":
        for path, labels in zip(image_path_list, output_labels):
            print("path: {}  | prediction: {} ".format(path, labels))
    else:
        for img, path, labels in zip(imgdata_list, image_path_list, output_labels):
            testimg = img.reshape(28, 28)
            plt.figure("Label: {}  |  Predicted Label: {}"
                       .format(path, labels))
            plt.imshow(testimg, cmap='gray')
        plt.show()


def runBatchMnist(mnistIndex, mode="rnn", display="text"):
    mnist = deepNet.mnist

    imgdata_list = []
    imgdata_labels = []
    for i in mnistIndex:
        imgdata = mnist.test.images[i].reshape(1, 784)
        imgdata_list.append(imgdata)
        label = mnist.test.labels[i]
        label = [i for i, l in enumerate(label) if l != 0]
        imgdata_labels.append(label)

    # Use recursive neural network
    # or multilayer perceptron neural network
    if mode.lower() == "rnn":
        output_labels = rnnNet.use_neural_network_list(imgdata_list)
    else:
        output_labels = deepNet.use_neural_network_list(imgdata_list)

    # Print in console or display with matplotlib
    if display.lower() == "text":
        for index, labels, olabels in zip(mnistIndex, imgdata_labels, output_labels):
            print("index: {}  | labels: {}  | prediction: {} ".format(index, labels, olabels))
    else:
        for index, labels, olabels, img in zip(mnistIndex, imgdata_labels, output_labels, imgdata_list):
            testimg = img.reshape(28, 28)
            plt.figure("ImageIndex: {}  |  Label: {}  |  Predicted Label: {}"
                       .format(index, labels, olabels))
            plt.imshow(testimg, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main()
