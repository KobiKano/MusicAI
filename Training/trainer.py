import numpy as np
import pandas as pd
import Neural.neural

# start training neural network

# initialize
# access csv
data = pd.read_csv("../Data/Songs/song_data.csv")
data = data.transpose()

arr = data.to_numpy()  # convert csv to 2D numpy array where first dimension is an individual song

# init neural network with 2 layers of size 100 and 5001 inputs and 10 outputs
network = Neural.neural.Network(2, 100, 5000, 11)

iteration = 0
for song in arr:  # each song is an array of song frequency domain with index 5001 as the class value
    # skip first
    if iteration == 0:
        iteration += 1
        continue

    # copy song into int array to check if problematic song
    arr_check = np.array(song, dtype=int)

    # check if majority values are zero
    if np.count_nonzero(arr_check == 0) == 0:
        # ignore song
        print(f"Ignoring Song: {iteration}")
        continue

    # feed song into neural network
    # isolate frequency array and class value
    inputs = song[0:len(song) - 2]
    desired = np.zeros(11, dtype=int)
    desired[int(song[5001])] = 1

    # feed formatted values into array
    print(f"Training Song: {iteration}")
    network.train(inputs, desired)

    iteration += 1

# save trained values
network.save_weights_biases("../Data/NeuralWeightsBiases/network.csv")
