import numpy as np
import pandas as pd
import Neural.neural
from tqdm import tqdm

# start training neural network

# initialize
# access csv
data = pd.read_csv("../Data/Songs/song_data.csv")
data = data.transpose()

arr = data.to_numpy()  # convert csv to 2D numpy array where first dimension is an individual song

# init neural network with 2 layers of size 100 and 5001 inputs and 10 outputs
network = Neural.neural.Network(2, 100, 5000, 11,
                                1.0, 10.0)
network.set_learning_rate(3.0)  # ***CAN SET LEARNING RATE***

# save random weights and biases for display in main program
network.save_weights_biases("../Data/NeuralWeightsBiases/random_network.csv")

num_success = 0
epoch_size = 100
epoch_num = 0
# each song is an array of song frequency domain with index 5001 as the class value
for iteration in tqdm(range(len(arr)), desc="Training..."):
    # skip first
    if iteration == 0:
        iteration += 1
        continue

    # check if next epoch and print accuracy
    if iteration % epoch_size == 0:
        epoch_num += 1
        print("Accuracy of Epoch {} : {}\n".format(epoch_num, float(num_success) / float(epoch_size)))
        num_success = 0

    # feed song into neural network
    # isolate frequency array and class value
    inputs = arr[iteration][0:len(arr[iteration]) - 2]

    # check if invalid values exist
    if np.isnan(np.sum(inputs)):
        # ignore song
        continue

    desired = np.zeros(11, dtype=float)
    desired[int(arr[iteration][5001])] = 1.0

    # normalize inputs between 0.0 and 1.0
    inputs = np.array(inputs)
    inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))

    # check if invalid values exist
    if np.isnan(np.sum(inputs)):
        # ignore song
        continue

    # feed formatted values into array
    if network.train(inputs.tolist(), desired.tolist(), 1):
        num_success += 1

    iteration += 1

# save trained values
print("Finished Training!\n")
network.save_weights_biases("../Data/NeuralWeightsBiases/network.csv")
