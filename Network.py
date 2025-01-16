import random
import json
import numpy as np 

# function for returning the sigmoid function of a value. Here z can be a vector or matrix and the np module would apply the sigmoid
# function individually to each entry of the vector / matrix. Sigmoid Functions in maths is defined as f(x) = 1/(1+e^-x)
sigmoid = lambda z : 1.0/(1.0 + np.exp(-z))

# function for returning the derivative of the sigmoid function of a value. Similar to the sigmoid function 
sigmoidDash = lambda x : sigmoid(x)*(1-sigmoid(x))   # we can also write sigmoidDash as sigmoid(x)*(1-sigmoid(x))

def softmax(x):return np.exp(x)/np.sum(np.exp(x))
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
        self.biases = [(np.random.randn(y,1)) for y in sizes[1:]]

        # similar to the biases, create a list of initial random weight matrices of y rows and x columns where x, is the number
        # of neurons in the previous layer up to the second last layer and y is the number of neurons in the next layer starting 
        # from the second layer. This Matrix layout is so that the weight at the i th row and j th column represents the weight 
        # of the j th neuron of the previous layer with whose output value the i th neuron of the current layer should multiply it with  
        self.weights = [(np.random.randn(y,x)) for x,y in zip(sizes [:-1] , sizes [1:])]

    # function to forward or "feed forward" the output of one layer of neurons of the network into the next layer after calculating it's
    # sigmoid function value
    def feedNextLayer(self , a):
        for w,b in zip(self.weights[:-1] , self.biases[:-1]):
           # taking the dot product/product of the input and weight vector/matrices and adding the bias matrix/vector 
           # before calculating its sigmoid velue to forward to the next layer 
            a = sigmoid(np.dot(w,a) + b)
        a=softmax(np.dot(self.weights[-1] , a) + self.biases[-1])
        return a 
    
    # function to perform Stochastic Gradient Descent on the network. Stochastic Gradient Descent is a Gradient Descent technique where we
    # calculate the average cost of derivative function on a batch of random inputs and then adjust the weights and biases according
    # to that. This technique can help in faster learning and lesser computation needs
    # by the neural network as we dont train on individual inputs but rather batches of inputs at the cost of some precision.
    # If the batch sizes are not too large then most of the accuracy will pertain.
    def stochasticDescent(self , training_data , epochs , mini_batch_size , rate , test_data = None):
        # training data is the list of tuples(x,y) where x is the training input and y is its desired output
        # epochs is the number of times the network trains and iterates through the complete training data 
        # mini_batch_size is the number of samples in each mini batch
        # rate is the learning rate of the gradient descent
        # test_data by default is none otherwise we test the model against it

        if(test_data): num_tests = len(test_data)

        # len_train_set is the length of the training data sets or the number of inputs given as training data 
        len_train_set = len(training_data)

        for j in range(epochs):

            #shuffling the training data
            random.shuffle(training_data)

            #dividing the training data in mini batches
            #mini batches is a list of lists of training data. Where, each sub-list is of length mini_batch_size
            mini_batches = [training_data [k:k+mini_batch_size] for k in range(0 , len_train_set , mini_batch_size)]

            # updating weights and biases for each mini batch

            for mini_batch in mini_batches : 
                
                self.updateValues(mini_batch , rate)

            # printing the accuracy after each iteration of the training data by testing on the test data (if any)
            
            if(test_data):
                
                random.shuffle(test_data)
                print("Epoch " , j+1 , ": " , self.checkAccuracy(test_data) , " / " , num_tests)
            
            else:

                print("Epoch complete")

    # function to update the  weights and biases of the neural network according to the given and desired results of the mini batch 
    # by help of call to the backpropagation function    
    def updateValues(self , mini_batch , rate):
        
        # creating a list of weights and biases that will store the sum of the small change or differential weights and biases for
        # each entry of the mini batch
        # np.zeros(shape) creates a list/matrix of given shape with all entries as 0
       
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        gradient_weight = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:

            # the change in bias and weight required will be sent as tuple by the backpropagation function
            delta_gradient_bias , delta_gradient_weight = self.backpropagate(x,y)

            #adding the delta gradient values to the gradient lists
            gradient_bias = [gb + dgb for gb,dgb in zip(gradient_bias , delta_gradient_bias)]
            gradient_weight = [gw + dgw for gw,dgw in zip(gradient_weight , delta_gradient_weight)]

        #updating the bias and weight lists
        self.biases = [b-(rate/len(mini_batch))*gb for b, gb in zip(self.biases , gradient_bias)]
        self.weights = [w-(rate/len(mini_batch))*gw for w , gw in zip(self.weights , gradient_weight)]
     
    def backpropagate(self , x , y):

        #creating lists to store the values of delta b and delta w
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        #Step 1: feed forwrd to find the error in result

        #first setting the input vector as the activation and adding it to the activations list
        activation = x
        activations = [x] 

        #creating a list of the z/input vector of each layer
        z_list = []
        for b , w in zip(self.biases[:-1] , self.weights[:-1]):
            #calculating the input z of the next layer
            z = np.dot(w,activation)+b 
            z_list.append(z)
            #calculating the activation of this layer & adding it to the activations list
            activation = sigmoid(z)
            activations.append(activation)
        z=np.dot(self.weights[-1],activation)+self.biases[-1]
        z_list.append(z)
        activation = softmax(z)
        activations.append(activation) 
        
        # Step 2: calculating error of cost function
        # calculating cost derivative of last layer predicted output versus actual result using equation 1
        # the derivative of the sigmoid function is used here instead of the softmax for mathematical ease and computational efficiency
        # as the derivative of the softmax function is a bit complex and the sigmoid function derivative is a good approximation , with a little
        # change in hyper parameters most of the incurred losses of accuracy of the network can be compensated
        delta_cost = self.costDerivative(activations[-1] , y) * sigmoidDash(z_list[-1])

        #Step 3: calculating the amount of change to be brought in biases and weights using eqns 2, 3 & 4
        delta_b[-1] = delta_cost
        delta_w[-1] = np.dot(delta_cost , activations[-2].transpose())

        for l in range(2 , self.numLayers):
            delta_cost = np.dot(self.weights[-l+1].transpose() , delta_cost)*sigmoidDash(z_list[-l])

            delta_b[-l] = delta_cost
            delta_w[-l] = np.dot(delta_cost , activations[-l-1].transpose())

        #Step 4: return the delta matrices to find the gradient 
        return (delta_b , delta_w)

    #function to calculate the cost derivative for a MSE cost function with respect to activation a
    def costDerivative(self , activation , y):
        return (y-activation)
    
    #function to check the accuracy of the network
    def checkAccuracy(self , test_data ):

        test_results = [(np.argmax(self.feedNextLayer(x)) ,y) for (x, y) in test_data]
            
        return sum(x.all() == y.all() for x,y in test_results)
    
    #function to save the network sizes, weights and biases
    def saveNetwork(self, filename):
        data = {"sizes":self.sizes,
                "weights":[w.tolist() for w in self.weights],
                "biases":[b.tolist() for b in self.biases]}
        with open(filename, 'w') as f:
            json.dump(data, f)

    #function to load the network sizes, weights and biases
    def loadNetwork(self,filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.sizes = data["sizes"]
            self.weights = [np.array(w) for w in data["weights"]]
            self.biases = [np.array(b) for b in data["biases"]]