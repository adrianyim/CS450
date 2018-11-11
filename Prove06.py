### Single-Layer Perceptron Code ###

# import numpy as np

# class Perceptron(object):
#     # Implements a perceptron network
#     def __init__(self, input_size, lr=1, epochs=100):
#         self.W = np.zeros(input_size+1)
#         # add one for bias
#         self.epochs = epochs
#         self.lr = lr
    
#     def activation_fn(self, x):
#         return 1 if x >= 0 else 0
 
#     def predict(self, x):
#         z = self.W.T.dot(x)
#         a = self.activation_fn(z)
#         return a
 
#     def fit(self, X, d):
#         for _ in range(self.epochs):
#             for i in range(d.shape[0]):
#                 x = np.insert(X[i], 0, 1)
#                 y = self.predict(x)
#                 e = d[i] - y
#                 self.W = self.W + self.lr * e * x

# if __name__ == '__main__':
#     #
#     X = np.array([
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
#     ])
#     #
#     d= np.array([0, 0, 0, 1])

#     perceptron = Perceptron(input_size=2)
#     perceptron.fit(X, d)
#     print(perceptron.W)

### Multiple-Layer Perceptron Code ###

from numpy import exp, array, random, dot

# Create some random weights in layer 1 and layer 2
class NeuronLayer():
    def __init__(self, number_of_nodes, number_of_inputs_per_node):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_node, number_of_nodes)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # A Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # A derivative of the Sigmoid function
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network through a process of trial and error, and adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for i in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (4 nodes, each with 3 inputs): ")
        print( self.layer1.synaptic_weights)
        print("    Layer 2 (1 node, with 4 inputs):")
        print( self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 nodes, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("Created some random synaptic weights: ")
    neural_network.print_weights()

    # Some training set data

    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    # neural_network.train(training_set_inputs, training_set_outputs, 60000)

    # print("New synaptic weights after training: ")
    # neural_network.print_weights()

    # # Test the neural network with a new situation.
    # print("A new situation [1, 1, 0] -> ?: ")
    # hidden_state, output = neural_network.think(array([1, 1, 0]))
    # print(output)