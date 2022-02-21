import scipy.io
import os
import numpy as np

classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

# save .mat files as .npy files for convenience
def extractFiles():
    datasetFilePath = os.getcwd() + "/dataset/WILLOW-ObjectClass/"
    print(datasetFilePath)

    for c in classes:
        files = os.listdir(datasetFilePath + c + "/")
        for f in files:
            if f.endswith(".mat"):
                npFilename = f.replace(".mat", ".npy")
                npFilepath = os.getcwd() + "/dataset/numpyFiles/" + c + "/"
                mat = scipy.io.loadmat(datasetFilePath + c + "/" + f)
                # print(mat)
                arr = mat['pts_coord']
                nparr = np.array(arr)
                # print(nparr.shape)
                np.save(npFilepath + npFilename, nparr)
                print("processed", f)

if __name__ == '__main__':
    extractFiles()
