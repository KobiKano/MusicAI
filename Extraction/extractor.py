# This file contains functionality to parse through csv form data sets and write frequency domain data into output

import pandas as pd
import pytube as pt
import Extraction.fourier
class Extractor:
    # defaults
    path = "Data/Training/train.csv"
    df = 0
    index = 0
    max_index = 0
    end = False


    def set_path(self, path):
        # set path on specific string
        self.path = path

    def start(self):
        # find file from given path
        self.df = pd.read_csv(self.path)
        self.max_index = len(self.df.index)

    def next(self):
        # check if next exists
        if self.index == self.max_index:
            self.end = True
            return

        # find next song
        artist_name = self.df.loc[self.index]['Artist Name']
        song_name = self.df.loc[self.index]['Track Name']
        song_class = self.df.loc[self.index]['Class']

        # increment index
        self.index += 1

        # find song using pytube
        search_name = "{} by {}"
        print("Searching for " + search_name.format(song_name, artist_name) + "\n")
        try:
            yt = pt.Search(search_name.format(song_name, artist_name)).results[0]  # take first result
            song = yt.streams.filter(only_audio=True).first()

            # extract sound data and parse into frequency domain signal
            file = song.download(output_path="../Data/Songs")
            return Extraction.fourier.transform(file, song_class)
        except:
            print("ERROR occurred with search moving to next!!!")
            return
