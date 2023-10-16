import Extraction.extractor
import Extraction.fourier
import pandas as pd
# This file will parse all song data and categories to feed into neural network
# initialize locals
extractor = Extraction.extractor.Extractor()
data = {}
index = 0

# download all files
print("Downloading training files!\n")
extractor.set_path("../Data/Training/train.csv")
extractor.start()
extractor.set_max_songs(2001)
while not extractor.end:
    # organize training data to save in csv
    data[index] = extractor.next()

    # save all data in csv
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf="../Data/Songs/song_data.csv")  # honestly this is terrible formatting, but I'll make it work
    index += 1  # Just keep going until stack overflow and then we use whatever data we got

print("!!!finished downloading training files!!!\n")
