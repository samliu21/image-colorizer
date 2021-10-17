# image-colorizer

A deep convolutional neural network that adds colour to grayscale landscape images! 🌈

Languages: Python, Tensorflow 🐍

<img src='example.png' style='height: 400px'>

## Model
The first half of the neural network consists of part of the famous VGG16 algorithm. Some of the later layers, which are used for classification, were removed. The latter half of the network is made of a series of 2D convolution, dropout, upsampling, and 2D convolution transpose layers. The model uses the Adam optimizer and the mean squared error loss function. It was trained over tens of thousands of landscape images.

A summary of the neural network can be found below:

```
______________________________________________________
Layer                Output Shape              
______________________________________________________
VGG16                (None, 7, 7, 512)
Conv2D               (None, 7, 7, 256)
Conv2DTranpose       (None, 14, 14, 16)
Conv2D               (None, 14, 14, 256)
Dropout              (None, 14, 14, 256)
UpSampling2D		     (None, 28, 28, 256)
Conv2D               (None, 28, 28, 128)
Dropout              (None, 28, 28, 128)
UpSampling2D		     (None, 56, 56, 128)
Conv2D               (None, 56, 56, 64)
UpSampling2D		     (None, 112, 112, 64)
Conv2D               (None, 112, 112, 32)
Conv2D               (None, 112, 112, 2)
UpSampling2D		     (None, 224, 224, 2)
______________________________________________________
```