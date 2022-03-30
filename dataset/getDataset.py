import os
import numpy as np

def getDataset(imageClasses):
    # print("hello")
    datasetFilePath = os.getcwd() + "/dataset/numpyFiles/"

    for c in imageClasses:
        filepath = datasetFilePath + c
        files = os.listdir(filepath)
        for f in files:
            nparr = np.load(filepath + "/" + f)
            print(nparr.shape)
            # row column coordinates for points of interest
            # exit()

if __name__ == '__main__':
    getDataset("Car")
