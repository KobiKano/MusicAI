import ast

import Neural.neural
import pandas as pd
import pytube as pt
import Extraction.fourier
import numpy as np
import matplotlib.pyplot as plt
import regex as re

def findGenre(network):
    song = input("Enter Song Name\n")
    print(f"Searching for {song}\n")
    inputs = []

    try:
        yt = pt.Search(song).results[0]  # take first result
        song = yt.streams.filter(only_audio=True).first()

        # extract sound data and parse into frequency domain signal
        file = song.download(output_path="../Data/Songs")
    except:
        print("ERROR occurred with search!!!\nReturning to Menu\n")
        return

    # extract fourier of mp4
    inputs = Extraction.fourier.transform(file, 0)
    inputs = inputs[0:len(inputs) - 2]

    # copy song into int array to check if problematic song
    arr_check = np.array(inputs, dtype=int)

    # check if majority values are zero
    if np.count_nonzero(arr_check == 0) == 0:
        # ignore song
        print("Song Download failed!!!\nReturning to Menu\n")
        return

    # find predicted genre
    outputs = network.forward_prop(inputs)
    genre = outputs.index(max(outputs))

    # print genre
    match genre:
        case 0:
            print("Genre: Acoustic/Folk\n")
        case 1:
            print("Genre: Alt Music\n")
        case 2:
            print("Genre: Blues\n")
        case 3:
            print("Genre: BollyWood\n")
        case 4:
            print("Genre: Country\n")
        case 5:
            print("Genre: Hip Hop\n")
        case 6:
            print("Genre: Indie Alt\n")
        case 7:
            print("Genre: Instrumental\n")
        case 8:
            print("Genre: Metal\n")
        case 9:
            print("Genre: Pop\n")
        case 10:
            print("Genre: Rock\n")

# this function displays some cool inner workings of the neural network
def dispGraphs(network, rand_network):
    print("Graph of weights at first node of original input layer:\n")
    plt.plot(rand_network.network[0][0].weights)
    plt.show()
    print("Graph of weights at first node of input layer:\n")
    plt.plot(network.network[0][0].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at 30th node of original input layer:\n")
    plt.plot(rand_network.network[0][29].weights)
    plt.show()
    print("Graph of weights at 30th node of input layer:\n")
    plt.plot(network.network[0][29].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at last node of original input layer:\n")
    plt.plot(rand_network.network[0][len(network.network[0]) - 1].weights)
    plt.show()
    print("Graph of weights at last node of input layer:\n")
    plt.plot(network.network[0][len(network.network[0]) - 1].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at first node of original hidden layer:\n")
    plt.plot(rand_network.network[1][0].weights)
    plt.show()
    print("Graph of weights at first node of hidden layer:\n")
    plt.plot(network.network[1][0].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at 30th node of original hidden layer:\n")
    plt.plot(rand_network.network[1][29].weights)
    plt.show()
    print("Graph of weights at 30th node of hidden layer:\n")
    plt.plot(network.network[1][29].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at last node of original hidden layer:\n")
    plt.plot(rand_network.network[1][len(network.network[1]) - 1].weights)
    plt.show()
    print("Graph of weights at last node of hidden layer:\n")
    plt.plot(network.network[1][len(network.network[1]) - 1].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at first node of original output layer\n")
    plt.plot(rand_network.network[2][0].weights)
    plt.show()
    print("Graph of weights at first node of output layer\n")
    plt.plot(network.network[2][0].weights)
    plt.show()
    input("Enter anything to continue")

    print("Graph of weights at last node of original output layer\n")
    plt.plot(rand_network.network[2][len(network.network[2]) - 1].weights)
    plt.show()
    print("Graph of weights at last node of output layer\n")
    plt.plot(network.network[2][len(network.network[2]) - 1].weights)
    plt.show()
    input("Enter anything to continue")

# main function to display project
if __name__ == '__main__':
    # initialize network and locals
    network = Neural.neural.Network(2, 100, 5000, 11)
    rand_network = Neural.neural.Network(2, 100, 5000, 11)

    # load weights and biases from training
    df = pd.read_csv("../Data/NeuralWeightsBiases/network.csv")
    rand_df = pd.read_csv("../Data/NeuralWeightsBiases/random_network.csv")

    # load data into neural network
    network.set_weights_biases(df["weights"].tolist(), df["bias"])
    rand_network.set_weights_biases(rand_df["weights"].tolist(), rand_df["bias"])


    # start looping to ask for user input
    while True:
        print("Enter Associated Number for Function:\n"
              "1) Find Song Genre\n"
              "2) Display Graphs\n"
              "3) Exit\n")
        user_in = int(input())

        match user_in:
            case 1:
                findGenre(network)
            case 2:
                dispGraphs(network, rand_network)
            case 3:
                exit(0)
            case default:
                print("Unpredicted input, exiting!")
                exit(1)

