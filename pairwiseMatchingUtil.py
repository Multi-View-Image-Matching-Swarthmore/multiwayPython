import numpy as np
import os
import scipy.io
from classes import pairwiseMatches
import tqdm


'''
Calculate Torus-like numbers
Helps with calculating pairing matrix
Inputs:
- num: raw number to figure out new coordinate
- max: maximum size of torus

Outputs:
- new coordinate, given the wrap around of torus
'''
def torusCoords(num, max):
    if num < 0 or num >= max:
        return num % max
    else:
        return num

'''
Run Graph Matching on a batch of images
Inputs:
- datapath: filepath to image class (ex. Car)
- viewList: list of hypercols files for class (files end with hypercols_kpts.mat)
- pairingMode: type of pairing mode for matches matrix (all, loop, neighbor)
- pairingParam: the radius of the neighborhood if pairing mode is loop or neighbor
- wEdge: type of linear/graph matching (0, 1, 2, ect)

Outputs:
- pairMatches: Numpy array of pairwiseMatches class, shape is (len(viewList), len(viewList))
'''
def runGraphMatchBatch(datapath,viewList,pairingMode, pairingParam,wEdge=0):

    # initalize variables
    numImages = len(viewList)
    # print("viewList:", viewList)
    # print("numImages", numImages)
    pairs = np.zeros((numImages, numImages)) # 0 = false, 1 = true

    pairingMode = pairingMode.lower()

    # calculate pairing matrix
    if pairingMode == "all": # matrix should be all ones
        print("All Pairing Mode")
        pairs = np.ones((numImages, numImages)).astype(int)

    elif pairingMode == "loop":
        print("Loop Pairing Mode")
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

    # Create pairMatches array
    pairMatches = np.empty((numImages, numImages), dtype=pairwiseMatches)
    # print(pairMatches.shape)
    # print(viewList)

    # Do graph mathcing with hypercols file
    for i in tqdm.trange(numImages):
        for j in tqdm.trange(i + 1, numImages):
            if pairs[i][j] == 1:
                # print("running graph matching")
                viewPair = (viewList[i], viewList[j]) # pair of images
                # print(i , j)
                matchData = graphMatch(viewPair, wEdge) # run graph matching on two images
                # print(matchData)
                matchData.filename = (viewList[i], viewList[j])
                pairMatches[i][j] = matchData; # add to array

    return pairMatches


'''
Run Graph Matching on two images
Inputs:
- viewPair: tuple of two hypercols filepaths
- (optional)methodGM: Method for solving graph matching (sm, rw)
- (optional)methodDisc: Method for discretization (greedy, hungry)
- (optional)wEdge: Weigth of rigidicity
- (optional)wAngle: no idea tbh
- (optional)kNNInit: number of candidate correpondences for each feature
- (optional)nMaxInit: total number of candidate correpondences after optimization
- (optional)thScore: threshold of matching scores below which the match is ignored
- (optional)thRatio: no idea, threshold ratio ?
- (optional)thDist: no idea, threshold distance ?

Output:
- mData: a pairwiseMatches class
'''
def graphMatch(viewPair, methodGM = 'rw', methodDisc = 'greedy', wEdge = 0, wAngle = 0.3, kNNInit = 50, nMaxInit = 5000, thScore = 0, thRatio = 1, thDist = 0):

    # obtain intiial matches
    # print("Looking for initial matches...")
    simScores, matchInds = initMatch(viewPair,kNNInit,nMaxInit,thScore,thRatio)

    # load preprocessed data into class
    matchInfo = matchInds

    # number of features
    viewMat1 = scipy.io.loadmat(viewPair[0])
    nFeature = viewMat1['nfeature'][0][0]

    if len(matchInfo) == 0: # if match list is empty, exit method
        print("matchInfo list empty")
        return pairwiseMatches([], 0, [], []) # should hopefully never come here

    # graph matching to introduce geometric constraint
    Xraw = []
    X = []
    if wEdge <= 0:
        Xraw = simScores.astype(np.float64)
    else:
        # construct graphs
        print("Constructing graphs...")


        # do some spooky math
        print(matchInfo.shape) # (100, 2)
        unique_features1, new_feat1 = np.unique(matchInfo[:,0], return_inverse=True)
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
    if methodDisc == "greedy": # do greedyMatch
        # print(mData.matchInfo)
        X = greedyMatch(matchInfo, Xraw)
        # print(mData.matchInfo)
        # exit()
    elif methodDisc == "hungry":
        # X = optimalMatch(mData.matchInfo, Xraw)
        print("Implementation not finished: hungry")
        exit()
    else:
        print("Unknown discretization method:", methodDisc)
        exit()

    # load calculate X and Xraw into class
    mData.Xraw = np.array(Xraw).astype(np.float32)
    mData.X = X

    mData = pairwiseMatches(matchInfo, nFeature, Xraw, X)

    return mData

'''
Peform inital matching
Inputs:
- viewPair: tuple of two hypercols filpaths
- kNNInit: k value for k nearest neighbors
- nMaxInit:
- threshScore:
- threshRatio:
Outputs:
- simScores: Numpy array of similarity scores
- matchInds: Numpy array match indices
'''
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

    # calculate similarity scores and match indices
    for i in range(desc1.shape[1]):
        dotprods = desc1[:, i]@desc2
        # print(dotprods)
        # print(desc1[:, i])
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

'''
Perform Greedy Matching
Inputs:
- match: numpy array of matches
- score: Xraw, similarity scores, 1D, horizontal
-(optional)nMax: kinda wack, convergence condition?
Outputs:
-
'''
# check if works for normal score arrays
# works for sprase matrices right now
def greedyMatch(match, score, nMax=np.inf):
    # flag = np.zeros((len(score), 1))
    flag = np.zeros((score.shape[0], 1))
    # print(flag.shape)
    # print(match.shape)
    # print(score.shape)

    count = 0
    score = np.copy(score)
    max_ind = np.argmax(score)
    max_value = np.amax(score)
    # print(max_value)
    while (max_value > np.NINF and count < nMax): # sus
        count += 1
        flag[max_ind] = 1
        # score[match[0,:] == match[0,max_ind]] = np.NINF
        # score[match[1,:] == match[1,max_ind]] = np.NINF
        # print(match[max_ind])
        # print(match)
        # print(match[:,1])
        # import pdb; pdb.set_trace();
        ind1 = np.where(match[0,:] == match[0][max_ind])[0][0]
        ind2 = np.where(match[1,:] == match[1][max_ind])[0][0]
        # print("sfds")
        # print(ind1)
        # print(ind2)
        score[ind1] = np.NINF
        score[ind2] = np.NINF

        max_ind = np.argmax(score)
        max_value = score[max_ind]
        import pdb; pdb.set_trace();

    # print(np.sum(flag))

    return flag

def main(): # test runGraphMatchBatch
    datapath = "Car"
    viewList = np.zeros((10,10))
    pairingMode = "neighbor"
    pairingParam = 3
    runGraphMatchBatch(datapath,viewList,pairingMode,pairingParam)

if __name__ == '__main__':
    main()
