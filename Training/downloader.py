import Extraction.extractor
import Extraction.fourier
import pandas as pd
# This file will parse all song data and categories to feed into neural network
# initialize locals
extractor = Extraction.extractor.Extractor()
training = []

# download all files
print("Downloading training files!\n")
extractor.set_path("../Data/Training/train.csv")
extractor.start()
while not extractor.end:
    training.append(extractor.next())

print("!!!finished downloading training files!!!\n")

# organize training data to save in csv
data = {}
for i in range(len(training)):
    # saving values in frequency array with last value always being song class value
    data[i] = training[i]

# save all data in csv
df = pd.DataFrame(data=data)
df.to_csv(path_or_buf="../Data/Songs/song_data.csv")
