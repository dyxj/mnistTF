# mnistTF - mnist Data and TensorFlow #
Purpose of this project is to build a neural network with TensorFlow(TF). 
The network was trained with **mnist dataset** to recognize hand written digits (0-9).
Numbers were written with paint and fed to the network, results are shown below.

## Prerequisites and Setup ##
* mnist datasets
* install packages using pip with requirements_gpu.txt.
    * tensorflow-gpu==1.0.1 requires cuda installation
    * alternatively the cpu version of tensorflow can be used instead
* Generate config.ini with configInit.py (modify configInit.py to point 
to the appropriate paths)  

## Results ##  
**Label** : Expected output | **MP** : multilayer perceptron | **RNN** : recurrent neural network
  
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 7 | 8 | 9 | blank |
| MP Output | 0 | 1 | 2 | 3 | 5 | 5 | 5 | 8 | 1 | 8 | 9 | 5 |
| RNN Output | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 7 | 8 | 9 | 1 |
| Image | ![Alt text](/hand/0test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/1test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/2test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/3test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/4test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/5test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/6test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/7test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/7atest.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/8test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/9test.png?at=master&fileviewer=file-view-default) | ![Alt text](/hand/blank.png?at=master&fileviewer=file-view-default) |  

Using the multilayer perceptron the accuracy obtained was 96.73%, the recurrent neural network 
however manage to score 98.87. Although 2% may seem small the results shows that it makes a big 
difference.

## Reference & Resources ##
* sentdex : https://pythonprogramming.net/
* mnist data : http://yann.lecun.com/exdb/mnist/
