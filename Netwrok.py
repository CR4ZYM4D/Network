import numpy as np 

#designing the Network class

class Network(object):
    
    def __init__(self,sizes) :

        #defining the number of layers in the network
        self.numLayers = len(sizes)

        #defining the number of neurons in the network where sizes is the list of number of layers with sizes[i]= number of neurons in layer i 
        self.sizes = sizes

        #create a list of matrices of y rows and 1 column of biases from the second layer or index 1 of the sizes list i.e.
        # 1. create an initial random bias for each neuron in the second layer onwards
        # 2. store the bias of the neuron of each layer in a list/vector
        # 3. store that list in another list at an index as that of the index of the neuron layer-1    
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # similar to the biases, create a list of initial random weight matrices of y rows and x columns where x, is the number
        # of neurons in the previous layer up to the second last layer and y is the number of neurons in the next layer starting 
        # from the second layer. This Matrix layout is so that the weight at the i th row and j th column represents the weight 
        # of the j th neuron of the previous layer with whose output value the i th neuron of the current layer should multiply it with  
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes [:-1] , sizes [1:])]

        

         