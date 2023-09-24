import matplotlib.pyplot as plt
from pydub import AudioSegment
from librosa import load
from scipy.fft import rfft
import numpy as np
import os

# take in mp4 file path and return array of frequency domain values from k = 0 to 300
def transform(file, song_class):
    # file = os.path.abspath(file)

    print("Extracting mp4 into wav")

    wav = "new.wav"
    sound = AudioSegment.from_file(file, format="mp4")
    sound.export(wav, format="wav")

    # parse new wav file into array
    arr, sr = load(wav)
    # print("Wav length is: {}\n Sampling rate is: {}", len(arr), sr)

    print("converting wav data into frequency domain signal")

    # use scipy to convert to frequency domain signal limit to size 10,000, normally around two million
    output = rfft(arr, n=10000)
    output = np.real(output)  # only considering real values

    # plot output for debugging
    # x = range(len(output))
    # plt.plot(x, output)
    # plt.show()

    # delete given file
    os.remove(file)
    os.remove(wav)
    print("finished converting to frequency domain")

    # return dict with class and frequency domain array
    return np.append(output, song_class)

