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
- Uses static learning rate and simple back propagation to transform 5000 input values to 11 outputs dictating music genre

# Usage
- Run main file
- Pick option
- If 1, input name of song and artist, ignore rendering errors and see if your program finds correct genre
- If 2, program will output some graphs of weights for specific nodes
- Otherwise exits

# If you want to walk through my process yourself
- Start with the downloader file in the Training directory, edit number of songs you want to download from dataset
- Then move to trainer file in the Training directory and adjust values based on preferences
- Finally, you can launch main and look at the graphs of the neural network to see what kind of result the training had

# Notes with project (What I Learned)
- Unfortunately this AI is not very good.
- It is likely that the fourier transform of a wav file is not enough to differentiate the genre of a song from one another
- I am not a music expert, or really familiar with this kind of material, however the neural network I have built is modular enough that an implementation for a number recognition program would not be very difficult, and I will quite possibly do that someday
- If you are reading this and are experienced with this field, I am very accepting of criticism and would love to know what I can improve upon!!!