import numpy as np
from numpy import random
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


# declaring variables
TRAINING_DATA_IMG_PATH = "extra_docs/"
ROI_COUNT = 7


def main():
    print("Building the Model to predict ROI")

    folder_list = os.listdir(TRAINING_DATA_IMG_PATH)
    for folder in folder_list:
        filename = TRAINING_DATA_IMG_PATH + folder
        country = folder.split('-')[0]
        training_set, test_set = buildCountryModel(filename, country)

        # applying Model

        mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
        X_training_set = training_set[0]
        Y_training_set = training_set[1]
        mlp.fit(X_training_set, Y_training_set)
        print(test_set)

def buildCountryModel(fileLocation, country):
    print(fileLocation, country)
    try:
        with open(fileLocation, 'r') as f:
            lines = f.readlines()[1:]  # Starts reading from second line
    except Exception as e:
        print("\n\nFile not found")
        print("Please place your file in the same directory or provide an absolute path")
        print("In the event you're using data.csv, please place it in the same directory as this program file")
        exit(0)

    type = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]
    data = []
    for x in range (0,len(lines)):
        currLine = lines[x]
        tokens = currLine.strip().split(",")
        for y in range(0, ROI_COUNT):
            inputVal = type[y]
            pointer = 0
            targetVal = tokens[pointer+2: pointer+9]
            if -1 not in targetVal:
                data.append((inputVal, targetVal))

    random.shuffle(data)
    partition = int(0.8 * np.shape(data)[0])
    training_data, test_data = data[:partition], data[partition:]

    return np.array(training_data), np.array(test_data)






if __name__ == "__main__":
    main()

