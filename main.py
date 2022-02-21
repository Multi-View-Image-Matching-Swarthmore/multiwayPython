import scipy.io
import os
import numpy as np
from prepareDataset import extractFiles

classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

def main():
    extractFiles()

if __name__ == '__main__':
    main()
