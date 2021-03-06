# image-colorizer

A deep convolutional neural network that turns grayscale landscape images into coloured ones! 🌈

Languages: Tensorflow, Numpy, Matplotlib, Python 🐍

<img src='test/example.png' style='height: 400px'>

## Data
The following Kaggle datasets were used:
- <a href='https://www.kaggle.com/arnaud58/landscape-pictures'>Dataset 1</a>
- <a href='https://www.kaggle.com/theblackmamba31/landscape-image-colorization'>Dataset 2</a>

## Model
The first part of the neural network uses the pretrained VGG16 network. The second part of the network is a series of 2D convolution, dropout, upsampling, and 2D convolution transpose layers. The model uses the Adam optimizer and the mean squared error loss function.

A summary of the neural network can be found below:

```
Receives grayscale input images of size (224, 224, 3), where the R, G, and B values are the same.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 7, 7, 256)         1179904   
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 16)        262160    
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 256)       37120     
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 256)       0         
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 128)       295040    
_________________________________________________________________
dropout_1 (Dropout)          (None, 28, 28, 128)       0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        73792     
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 112, 112, 32)      18464     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 112, 112, 2)       578       
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 224, 224, 2)       0         
=================================================================
Total params: 16,581,746
Trainable params: 1,867,058
Non-trainable params: 14,714,688
_________________________________________________________________
```
