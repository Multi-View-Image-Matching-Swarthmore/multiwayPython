import numpy as np

def torusCoords(num, max):
    if num < 0 or num >= max:
        return num % max
    else:
        return num


# (numpy files,hypercols_kpts,'all',[],'wEdge', 0)
def runGraphMatchBatch(datapath,viewList,pairingMode,pairingParam,varargin):
    #     % Options:
    # % pairingMode = 'all' | 'loop' | 'neighbor'
    # % pairingParam = ...
    # % the radius of neighnorhood, if pairingMode == 'loop' and 'neighbor'
    # % a binary matrix indicating the pairs to match, if pairingMode == 'userinput'

    numImages = len(viewList)
    pairs = np.zeros((numImages, numImages)) # 0 = false, 1 = true

    pairingMode = pairingMode.lower()

    if pairingMode == "all":
        print("All Pairing Mode")
        pairs = np.ones((numImages, numImages)).astype(int)
    elif pairingMode == "loop":
        print("Loop Pairing Mode")
        # make pairingParam neighbors
        for i in range(pairs.shape[0]):
            for n in range(pairingParam):
                dist = n + 1
                neighborIndex = torusCoords(i + dist, numImages)
                pairs[i][neighborIndex] = 1

        pairsT = np.transpose(pairs)
        pairs = np.logical_or(pairs, pairsT).astype(int)
    elif pairingMode == "neighbor":
        print("Neighbor Pairing Mode")
        for i in range(pairs.shape[0]):
            for n in range(pairingParam):
                dist = n + 1
                neighborIndex1 = i + dist
                neighborIndex2 = i - dist
                if neighborIndex1 < numImages:
                    pairs[i][neighborIndex1] = 1
                if neighborIndex2 >= 0:
                    pairs[i][neighborIndex2] = 1
        pairsT = np.transpose(pairs)
        pairs = np.logical_or(pairs, pairsT).astype(int)
    else:
        print("Invalid pairing mode:", pairingMode)
        exit()

    print(pairs)

def main():
    datapath = "asdfghjkl"
    viewList = np.zeros((10,10))
    pairingMode = "neighbor"
    pairingParam = 1
    varargin = None
    runGraphMatchBatch(datapath,viewList,pairingMode,pairingParam,varargin)

if __name__ == '__main__':
    main()
