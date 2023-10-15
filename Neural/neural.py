import pandas as pd
import Neural.node as node
import random as rand
import math


# This file contains all functions for the initialization of a neural network

# random filler to allow negative values
def fill(num):
    output = []
    for i in range(num):
        output.append(rand.randint(0, 50) - 50)
    return output


# this is the network class that will house all functions for the neural network
class Network:
    network = []
    layers = 0

    def __init__(self, num_layers, layer_size, num_inputs, num_outputs):
        # initialize layers
        self.layers = num_layers
        for i in range(num_layers):
            if i == 0:
                self.network.append([node.Node(fill(num_inputs), rand.randint(0, 100) - 50)] * layer_size)
                continue
            self.network.append([node.Node(fill(layer_size), rand.randint(0, 100) - 50)] * layer_size)

        # initialize output layer
        self.network.append([node.Node(fill(layer_size), rand.randint(0, 100) - 50)] * num_outputs)

        print(len(self.network))
        print(len(self.network[self.layers]))

    def set_weights_biases(self, weights, biases):
        # assume weights are 3-dimensional array
        for i in range(len(weights)):  # layer
            for j in range(len(weights[i])):  # node
                self.network[i][j].set_weights(weights[i][j])
                self.network[i][j].set_bias(biases[i][j])

    def save_weights_biases(self, path):
        data = {
            "layer": [],
            "node": [],
            "weights": [],
            "bias": []
        }
        # save weights and biases in csv
        for i in range(len(self.network)):  # layer
            for j in range(len(self.network[i])):  # node
                data["layer"].append(i)
                data["node"].append(j)
                data["weights"].append(self.network[i][j].weights)
                data["bias"].append(self.network[i][j].bias)
        df = pd.DataFrame(data=data)
        df.to_csv(path)

    def train(self, inputs, desired):
        # train on array of inputs
        outputs = self.forward_prop(inputs)
        # check outputs and compare to desired cost
        cost = 0
        for i in range(len(outputs)):
            cost += math.pow((outputs[i] - desired[i]), 2)

        print("Cost of last train: " + str(cost))
        # send cost feedback backwards to adjust weights and biases
        self.backward_prop(desired, inputs)

    def forward_prop(self, inputs):
        outputs = []
        # go through each node in first layer and feed inputs and save outputs
        for j in range(len(self.network[0])):
            outputs.append(self.network[0][j].get_sum(inputs))

        # go through each node in each following layer and find outputs
        for i in range(1, self.layers + 1):
            next_outputs = []
            # go through each node and feed inputs and save outputs
            for j in range(len(self.network[i])):
                next_outputs.append(self.network[i][j].get_sum(outputs))
            outputs = next_outputs.copy()

        return outputs

    def backward_prop(self, desired, inputs):
        # find gradients at output layer
        prev_gradients = []
        for i in range(len(self.network[self.layers])):
            local_gradient = 2*(self.network[self.layers][i].last_sum - desired[i])
            sigmoid_prime = self.network[self.layers][i].sigmoid_prime  # used for dynamic learning rate
            prev_gradients.append(local_gradient)

            # find new weights and biases
            for j in range(len(self.network[self.layers][i].weights)):
                self.network[self.layers][i].weights[j] -= local_gradient * sigmoid_prime * self.network[self.layers - 1][j].last_sum

            self.network[self.layers][i].bias -= local_gradient * sigmoid_prime

        # find gradients of lower layers
        for i in range(self.layers - 1, 0):
            gradients = []
            for j in range(len(self.network[i])):
                # find local gradient by summing over values in L + 1 layer
                local_gradient = 0
                print(len(self.network[i + 1]))
                for k in range(len(self.network[i + 1])):
                    # add product of weight from this node to every node in L + 1
                    #                sigmoid prime of other node
                    #                calculated local gradient at other node
                    local_gradient += self.network[i + 1][k].weights[j] * self.network[i + 1][k].sigmoid_prime * prev_gradients[k]

                gradients.append(local_gradient)

                # find dynamic learning rate
                sigmoid_prime = self.network[i][j].sigmoid_prime

                # find new weights and biases
                # check if on lowest layer
                if i == 0:
                    for k in range(len(self.network[j].weights)):
                        self.network[i][j].weights[k] -= local_gradient * sigmoid_prime * inputs[k]

                else:
                    for k in range(len(self.network[j].weights)):
                        self.network[i][j].weights[k] -= local_gradient * sigmoid_prime * self.network[j - 1][k].last_sum

                self.network[i][j].bias -= local_gradient * sigmoid_prime

            prev_gradients = gradients.copy()

        return
