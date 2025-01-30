import Network
import numpy
import MNISTLoader as ML

training_data , validation_data , test_data = ML.loadWrapper()

network = Network.Network([784,120,10])


network.loadNetwork("network data.json")

#network.stochasticDescent(training_data,20,10,2,validation_data)

# print("Test Data Run Score: ",network.checkAccuracy(test_data)," / ",len(test_data))
# print("Validation Data Run Score: ",network.checkAccuracy(validation_data)," / ",len(validation_data))
# print("Training Data Run Score: ",network.checkAccuracy(training_data)," / ",len(training_data))

img = input("Enter the image path you want to read: ")

print(network.read(img))

network.saveNetwork("network data.json") 