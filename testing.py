import Network
import random 
import MNISTLoader as ML

training_data , validation_data , test_data = ML.loadWrapper()

random.shuffle(test_data)

random.shuffle(validation_data)

network = Network.Network([784,120,10])

network.loadNetwork("network data.json")

network.stochasticDescent(training_data,10,10,12,validation_data)

print("Test Data Run Score: ",network.checkAccuracy(test_data)," / 10000")

print("Training Data Run Score: ",network.checkAccuracy(training_data)," / 50000")

network.saveNetwork("network data.json")