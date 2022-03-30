import numpy as np
import os
import scipy.io
from classes import pairwiseMatches
import tqdm

def torusCoords(num, max):
    if num < 0 or num >= max:
        return num % max
    else:
        return num


# (numpy files,hypercols_kpts,'all',[],'wEdge', 0)
# datapath = class
# viewlist = hypercols files in class
def runGraphMatchBatch(datapath,viewList,pairingMode,pairingParam):
    #     % Options:
    # % pairingMode = 'all' | 'loop' | 'neighbor'
    # % pairingParam = ...
    # % the radius of neighnorhood, if pairingMode == 'loop' and 'neighbor'
    # % a binary matrix indicating the pairs to match, if pairingMode == 'userinput'

    numImages = len(viewList)
    # print("viewList:", viewList)
    # print("numImages", numImages)
    pairs = np.zeros((numImages, numImages)) # 0 = false, 1 = true

    pairingMode = pairingMode.lower()

    if pairingMode == "all":
        print("All Pairing Mode")
        pairs = np.ones((numImages, numImages)).astype(int)
        # TODO: do i need to remove self matches, along diagonal?

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

    # load hypercols file
    # views = [] # array of hypercols files
    # for f in files:
    #     if f.endswith(".hypercols_kpts.mat"):
    #         views.append(datasetFilePath + datapath + "/" + f)

    pairMatches = np.empty((numImages, numImages), dtype=pairwiseMatches)
    # print(pairMatches.shape)
    # print(viewList)
    #graph mathcing with hypercols file
    for i in tqdm.trange(numImages):
        for j in tqdm.trange(i + 1, numImages):
            if pairs[i][j] == 1:
                # print("running graph matching")
                viewPair = (viewList[i], viewList[j])
                # print(i , j)
                matchData = graphMatch(viewPair)
                # print(matchData)
                matchData.filename = (viewList[i], viewList[j])
                # fix print statement
                # print(f'Done! (%d,%d) feautres, %d initial matches, %d output matches\n',matchData.nFeature,matchData.nFeature,len(matchData.X),sum(matchData.X))
                # import pdb;  pdb.set_trace()
                pairMatches[i][j] = matchData;

    return pairMatches


# remember to add documentation again for params
def graphMatch(viewPair, methodGM = 'rw', methodDisc = 'greedy', wEdge = 0, wAngle = 0.3, kNNInit = 50, nMaxInit = 5000, thScore = 0, thRatio = 1, thDist = 0):
    # obtain intiial matches
    # print("Looking for initial matches...")
    simScores, matchInds = initMatch(viewPair,kNNInit,nMaxInit,thScore,thRatio)
    # print("simScore---")
    # print(simScores)
    # print("Match Inds---")
    # print(matchInds)

    # viewMat2 = scipy.io.loadmat(viewPair[1])
    # print(viewMat1['nfeature'])
    # exit()

    # load preprocessed data into class
    mData = pairwiseMatches()
    mData.matchInfo = matchInds

    # number of features
    viewMat1 = scipy.io.loadmat(viewPair[0])
    mData.nFeature = viewMat1['nfeature'][0][0]

    if len(mData.matchInfo) == 0: # if match list is empty, exit method
        return mData

    # graph matching to introduce geometric constraint
    Xraw = []
    X = []
    if wEdge <= 0:
        Xraw = simScores.astype(np.float64)
    else:
        # construct graphs
        print("Constructing graphs...")


        # do some spooky math
        print(mData.matchInfo.shape) # 100, 2
        unique_features1, new_feat1 = np.unique(mData.matchInfo[:,0], return_inverse=True)
        print(unique_features1)
        print(new_feat1)
        # increasing order
        # ia = index of unique values in original array
        # ic = original list in terms of unique array
        # https://www.mathworks.com/matlabcentral/answers/327742-how-does-the-command-unique-work
        print("Implementation not finished, sorry!")
        exit()


        # choose graph matching methodDisc
        if methodGM == "sm":
            print("spectral matching")
        elif methodGM == "rw":
            print("random walk")
        else:
            print("Unkown graph matching method:", methodGM)
            exit()
    # discretization
    # print("discretization...")
    if methodDisc == "greedy":
        X = greedyMatch(mData.matchInfo, Xraw)
    elif methodDisc == "hungary": #spelling ?
        # X = optimalMatch(mData.matchInfo, Xraw)
        print("Implementation not finished: hungary")
    else:
        print("Unknown discretization method:", methodDisc)
        exit()

    mData.Xraw = np.array(Xraw).astype(np.float32)
    mData.X = X

    return mData


def initMatch(viewPair,kNNInit,nMaxInit,threshScore,threshRatio):

    # loading in view hypercols
    viewMat1 = scipy.io.loadmat(viewPair[0])
    viewMat2 = scipy.io.loadmat(viewPair[1])
    # getting descriptions
    desc1 = viewMat1['desc'].astype(np.float32)
    desc2 = viewMat2['desc'].astype(np.float32)
    # normalize along cols
    desc1 /= np.linalg.norm(desc1, axis=0)
    desc2 /= np.linalg.norm(desc2, axis=0)
    # getting feature locations (2x10)
    featLocs2 = viewMat2['frame']
    # print(frame2)

    matchInds = []
    simScores = []

    # recompute threshold distance (scale to image size?)
    # threshDist = np.max(viewMat2['img'].shape)*threshDist

    for i in range(desc1.shape[1]):
        dotprods = desc1[:, i]@desc2
        # print(dotprods)
        # print(desc1[:, i])
        # exit()
        sortedInds = np.argsort(-dotprods) # grab indicies of sorted array
        frame2 = featLocs2[:,sortedInds] # same order as sortedinds
        dotprods = dotprods[sortedInds] # sort by descending order
        if dotprods[0]/(dotprods[1] + 1e-10) < threshRatio: # threshold ratio test
            continue
        c = 0
        for j,val in enumerate(dotprods):
            if val <= threshScore: # score thresholding
                continue
            c += 1
            matchInds.append((i, sortedInds[j])) # append match indicies
            simScores.append(val) # append similarity score
            # in original, used threshold distancing to remove points too close together
            # seems kinda useless with so little points
            # dist = np.sqrt(np.sum((frame2 - frame2[j])**2, axis=0)) # column-wise norm
            if c >= kNNInit:
                break
    matchInds = np.array(matchInds)
    simScores = np.array(simScores)
    if simScores.shape[0] > nMaxInit:
        simSortInds = np.argsort(-simScores)
        simScores = simScores[:nMaxInit]
        matchInds = matchInds[simSortInds[:nMaxInit]]

    matchInds = matchInds.astype(np.uint16)

    return (simScores, matchInds)

# greedy selection
def greedyMatch(match, score, nMax=np.inf):
    flag = np.zeros((len(score), 1))
    # print(flag.shape)
    # print(match.shape)
    # print(score.shape)

    count = 0
    max_ind = np.argmax(score)
    max_value = score[max_ind]
    # print(max_value)
    while max_value > np.NINF and count < nMax: # sus
        count += 1
        flag[max_ind] = 1
        # score[match[0,:] == match[0,max_ind]] = np.NINF
        # score[match[1,:] == match[1,max_ind]] = np.NINF
        # print(match[max_ind])
        # print(match)
        # print(match[:,1])
        ind1 = np.where(match[:,0] == match[max_ind][0])
        ind2 = np.where(match[:,1] == match[max_ind][1])
        # print("sfds")
        # print(ind1)
        # print(ind2)
        score[ind1] = np.NINF
        score[ind2] = np.NINF

        max_ind = np.argmax(score)
        max_value = score[max_ind]

    # print(np.sum(flag))

    return flag

def main():
    datapath = "Car"
    viewList = np.zeros((10,10))
    pairingMode = "neighbor"
    pairingParam = 3
    runGraphMatchBatch(datapath,viewList,pairingMode,pairingParam)

if __name__ == '__main__':
    main()
