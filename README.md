A simple 3 layer feed forward neural network that employs the stochastic gradient descent and backpropagation algorithms 
along with the sigmoid activation function and L2 regularization techniques to recognize handwritten images from the MNIST dataset 
with an accuracy of about 98.5% written in python3.13. 

The cost function is a cross entropy error function and consists of 120 hidden neurons

The images are 50x50 pixel digits written by me in paint, of which the network is able succesfully able to recognize at least 5 of namely 0,1,4,5 and 7 the failure in the other 5 digits is most likely due to loss of quality in resizing and considering how refined the MNIST dataset is made to be (the images are actually about 20 or 22 pixels each side if I remember correctly and 2 to 4 pixels padding on each side)
