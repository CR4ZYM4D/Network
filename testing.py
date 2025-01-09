import Network
import MNISTLoader as ML

training_data , validation_data , test_data = ML.loadWrapper()

network = Network.Network([784 , 100 , 10])

network.stochasticDescent(training_data,20,20,3,validation_data)

print(network.checkAccuracy(test_data))