
#downloaded libraries
import tensorflow as tf
import keras
import numpy as np 
from keras import layers


#loading the MNIST dataset

(training_data , training_answer),(testing_data , testing_answer) = keras.datasets.mnist.load_data()

training_data = training_data.astype("float32")/255.0

testing_data = testing_data.astype("float32")/255.0

training_data = np.expand_dims(training_data,-1)

testing_data = np.expand_dims(testing_data,-1)


#defining the network 

model = keras.Sequential(

        [

        layers.Conv2D(32 , kernel_size=(3,3) , activation="relu" , input_shape = (28,28,1)), #Convolution layer
        layers.MaxPooling2D(pool_size=(2,2)),                                                #Max Pooling Layer
        layers.Conv2D(64 , kernel_size=(3,3) , activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),                                                                    #Converting to Vector
        layers.Dense(50 , activation="sigmoid"),                                             #sigmoid layer
        layers.Dropout(0.5),                                                                 #dropout for overfitting
        layers.Dense(10 , activation = "softmax")                                            #softmax for probabilistic results

        ]

        )

model.compile(loss="sparse_categorical_crossentropy" , optimizer="adam" , metrics=["accuracy"])

model.fit(training_data , training_answer , batch_size=10, epochs=5 , validation_split=0.2)
