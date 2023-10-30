# Music AI
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img alt="GitHub contributors" src="https://img.shields.io/github/contributors/KobiKano/MusicAI?color=green">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KobiKano/MusicAI?color=blue">

# Project Description
The goal of this project was to push myself into machine learning.  I wanted to gain a deeper understanding of neural networks as well as increase my proficiency in python, as I have never coded in python before

# Features
- Uses pytube to find audio data for any song user searches
- Feeds frequency domain transformation of audio into neural network
- Outputs genre of song

# Fourier Extraction
- Uses pytube, numpy and, scipy to extract fourier transform signal from csv
- CSV provided by https://www.kaggle.com/datasets/purumalgi/music-genre-classification

# Neural Network
- Uses dynamic learning rate and simple back propagation to transform 5000 input values to 11 outputs dictating music genre

# Usage
- Run main file
- Pick option
- If 1, input name of song and artist, ignore rendering errors and see if your program finds correct genre
- If 2, program will output some graphs of weights for specific nodes
- Otherwise exits