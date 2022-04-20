import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import time
import sys

import scipy.io

from pairwiseMatchingUtil import greedyMatch
from classes import pairwiseMatches, jointMatchInfo

#runJointMatch(pMatch,C,'Method','pg','univsize',10, 'rank',3,'lambda', 1);
# add documentation again
def runJointMatch(pMatch, C, method='pg', univsize=10, rank=3, l=1):
    nFeature = np.zeros((pMatch.shape[0], 1))
    # print(nFeature.shape)
    filename = np.empty((nFeature.shape), dtype=str)
    # print(filename.shape)

    nMatches = 0

    for i in range(pMatch.shape[0]):
        for j in range(i+ 1, pMatch.shape[1]):
            # checking if fields are empty
            if len(pMatch[i][j].X) == 0:
                print("X is empty for:", i, j)
                exit()
            if pMatch[i][j].nFeature == 0:
                print("nFeature is empty for:", i, j)
                exit()
            if len(pMatch[i][j].filename) == 0:
                print("filename is empty for:", i, j)
                exit()

            nMatches += np.sum(pMatch[i][j].X)
            nFeature[i] = pMatch[i][j].nFeature
            nFeature[j] = pMatch[i][j].nFeature
            filename[i] = pMatch[i][j].filename[0]
            filename[j] = pMatch[i][j].filename[1]

    # TODO:  line 66 in runJoinMatch
    # print(nFeature.shape)
    # exit()
    nFeatureWithZero = np.insert(nFeature, 0, 0)
    cumulativeIndex = np.cumsum(nFeatureWithZero).astype(int)
    # print(cumulativeIndex)\

    # print(pMatch)
    pDim = pMatch.shape[0] # should be the same as pMatch.shape[1]
    globalNumPoints = (pDim*(pDim-1)//2) # number of images pair combinations
    globalNumPoints *= pMatch[0][1].matchInfo.shape[0] # number of feature per image
    # pMatch is upper triangular btw
    # print(globalNumPoints)
    # exit()
    ind1 = np.zeros((1, globalNumPoints))
    ind2 = np.zeros((1, globalNumPoints))
    flag = np.zeros((1, globalNumPoints))
    score = np.zeros((1, globalNumPoints))
    z = 0

    # print(pMatch.shape[0])
    #
    # exit()

    for i in range(pDim): # check
        for j in range(i+1, pDim):
            # try:
            matchList = pMatch[i][j].matchInfo.astype(np.float64) # 100 x 2
            # print(matchList.shape)
            n = int(pMatch[i][j].X.shape[0])
            # print("n:", n)
            # print(z, z+n)

            # print("matchList",matchList[:,0])
            # ind1[z:z+n] = matchList[0,:] + cumulativeIndex[i] # check index?
            # print("a:", a.shape)
            # print(ind1.shape)
            # print(ind1[0, z:z+n].shape)
            # print(z, z+n)
            ind1[0, z:z+n] = matchList[:,0] + cumulativeIndex[i]
            ind2[0, z:z+n] = matchList[:,1] + cumulativeIndex[j]

            flag[0, z:z+n] = pMatch[i][j].X.reshape((1, -1))

            # print(pMatch[i][j].Xraw)
            score[0, z:z+n] = normGrayscale(pMatch[i][j].Xraw)
            # why are we grayscaling the Xraw? use opencv? normalizing data?
            # print(score[0, z:z+n][-2])
            # exit()
            # import pdb;  pdb.set_trace()
            z += n
            # exit()
            # except:
            #     print("error! oh no")
            #     exit()
    ind1 = ind1.reshape(-1,).astype(int)
    ind2 = ind2.reshape(-1,).astype(int)
    score = score.reshape(-1,)
    flag = flag.reshape(-1,)
    # print(ind2.shape)
    # print(score.shape)
    # print(cumulativeIndex[cumulativeIndex.shape[0] - 1])
    # exit()
    # original scores
    # M = sparse(ind1,ind2,score,cumIndex(end),cumIndex(end));
    # row, col, value, height of matrix, width of matrix
    lastIndex = int(cumulativeIndex[cumulativeIndex.shape[0] - 1])
    # print(lastIndex)
    # exit()
    M = csr_matrix((score, (ind1, ind2)), shape=(lastIndex, lastIndex))
    # print(M.shape)
    # exit()
    M = M + M.T
    # print(M.shape)
    # exit()

    # binary scores
    # Mbin = sparse(ind1,ind2,flag,cumIndex(end),cumIndex(end));
    Mbin = csr_matrix((flag, (ind1, ind2)), shape=(lastIndex, lastIndex))
    Mbin = Mbin + Mbin.T
    # print(Mbin.shape)
    # exit()
    vM = Mbin

    Size = min(univsize, min(nFeature))
    # print(Size)
    # exit()
    Z = []
    print("Running join match, problem size = (" + str(vM.shape[0]) + "," + str(Size) + ")")
    # exit()

    method = "spectral" # debugging line, shoudl eventually remove -ere

    if method == "spectral":
        print("Spectral Matching...")
        M_out, eigV, tInfo = spectralMatch(vM, nFeature, Size)
    elif method == "matchlift":
        print("matchlift (MatchLift) not implemented! Sorry!")
        exit()
    elif method == "als":
        pprint("als (MatchALS) not implemented! Sorry!")
        exit()
    elif method == "pg":
        print("pg (proposed method) not implemented! Sorry!")
        exit()
    else:
        print("Unkown Multi-Object Matching method:", method)
        exit()

    # check M_out
    # M_out is wrong !!!
    mat = scipy.io.loadmat("M_out.mat")
    matlabM_out = np.array(mat['M_out'])

    compare = matlabM_out == M_out

    print(matlabM_out.shape)
    print(M_out.shape)
    print(np.sum(matlabM_out))
    print(np.sum(M_out))
    print(compare.all())

    exit()

    for r in range(400):
        for c in range(400):
            if not compare[r][c]:
                print("not equal: ", r, c)
                print("Matlab:", matlabM_out[r][c])
                print("Python:", M_out[r][c])
                exit()
    exit()

    # output/save files?
    jMatch = np.empty((pDim, pDim), dtype=pairwiseMatches)

    for i in range(pDim): # check
        for j in range(i+1, pDim):
            print()
            # [ind1,ind2] = find(   M_out( csum(i)+1:csum(i+1) , csum(j)+1:csum(j+1) )   );
            # return row, col of the nonzero elements indices
            rowStart = cumulativeIndex[i] + 1
            rowStop = cumulativeIndex[i+1]
            colStart = cumulativeIndex[j] + 1
            colStop = cumulativeIndex[j+1]

            print(rowStart, rowStop)
            print(colStart, colStop)

            subMatrix = M_out[rowStart:rowStop, colStart:colStop]
            ind1, ind2 = np.nonzero(subMatrix)

            print(ind1.shape)
            print(ind1)

            exit()

            # if ind1 is not empty aka has nonzero elements
            # if matrix has nonzero elements
            if ind1.shape[0] > 0:
                # remove conflict
                # Xraw = vM(csum(i)+1:csum(i+1),csum(j)+1:csum(j+1));
                Xraw = vM[rowStart:rowStop, colStart:colStop]

                # # TODO: help
                # Xraw = Xraw(sub2ind(size(Xraw),ind1,ind2));
                # https://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
                Xraw = Xraw

                # X = greedyMatch([ind1';ind2'],Xraw);
                # def greedyMatch(match, score, nMax=np.inf):
                arr = np.concatenate(ind1.T, ind2.T, axis=1) # double check axis?
                X = greedyMatch(arr, Xraw)

                # store results
                # jMatch(i,j).matchInfo.match = [ind1';ind2'];
                # jMatch(i,j).X = X;
                # jMatch(i,j).Xraw = Xraw;
                # jMatch(i,j).filename = [filename(i),filename(j)];
                f = np.array([filename[i], filename[j]])

                # jMatch(i,j).nFeature = [nFeature(i),nFeature(j)];
                nf = np.array([nFeature[i], nFeature[j]])

                jMatch[i][j] = pairwiseMatches(arr, nf, Xraw, X)
                jMatch[i][j].filename = f

    # jmInfo.eigV = eigV;
    # jmInfo.nFeature = csum;
    # jmInfo.filename = filename;
    # jmInfo.time = timeInfo.time;
    # jmInfo.Z = Z;
    jmInfo = jointMatchInfo(eigV, cumulativeIndex, filename, tInfo, Z)

    print("end")
    exit()

    return jMatch,jmInfo,tInfo

# [M_out,eigV,timeInfo] = mmatch_spectral(vM,nFeature,Size);
def spectralMatch(W, nFeature, universeSize):
    # exit()
    k = min(universeSize, W.shape[0])
    start = time.time()
    n = W.shape[0]
    print(n, k, universeSize)
    # print(W.shape)
    # w, V = eigh(W, eigvals=(n-k, n-1))
    w, V = eigs(W, k=k,which="LM") # w = k eigenvalues, V = k eigenvectors
    print(V.shape)
    # print(V)
    print(w.shape)
    # print(w)
    # exit()
    ## TODO: what are x and y
    # print()
    # exit(V[:, :k] == V)

    #NOTE: V[:, :k] is the same as V
    V = np.real(V)
    Y = rounding(V, nFeature, threshold=0.5).astype(np.float32)
    # check Y
    mat = scipy.io.loadmat("Y.mat")
    matlabY = np.array(mat['Y'])

    compare = matlabY == Y

    # why are they different sizes
    print(matlabY.shape)
    print(Y.shape)
    # print(np.sum(matlabM_out))
    # print(np.sum(M_out))
    print(compare)
    exit()
    # print(Y)
    # exit()
    X = np.matmul(Y,Y.T)
    end = time.time()

    runtime = end - start
    print(runtime)
    # exit()

    return X, V, runtime

#Y = rounding(V(:,1:k),dimGroup,0.5);
def rounding(A, dimGroup, threshold=0.5): # can we just run k means???
    # normalize in order to calculate correlation coefficient of points
    A = norm_rows(A)
    heightA, widthA = A.shape

    # cumulative sum
    N = np.cumsum(dimGroup).astype(int)
    print("A:",A.shape)
    # print("N:", N)

    flag = np.zeros((heightA,1)) # indicates in row already has been assigned a max
    Y = np.zeros((heightA, widthA)) # ends up 400 x 16?
    p = -1
    scores = np.zeros((10))

    # print("Y shape:",Y.shape)
    # print("dim group shape",dimGroup.shape)

    # print("---")
    for i in range(heightA): # every row in A
        # print("i:", i)
        if flag[i] == 0:
            p += 1 # column we are on
            if p >= Y.shape[1]: # if we need more columns, add a column
                Y = np.concatenate([Y, np.zeros((Y.shape[0], 1), dtype=Y.dtype)],-1)
            Y[i][p] = 1
            flag[i] = 1
            # np.where N>= i, returns group greater than index
            # ob = find(N>=i,1,'first');
            ob1 = np.where(N > i)[0][:-1] # indices of all the groups after i
            # -1 needed ???
            # exit()
            currentRow = A[i]
            # print("\tcurrent row index:", i)
            # print("current row:",currentRow)
            # print("\t", ob1)
            # print("\tnum groups", ob1.shape)
            # exit()
            for groupIndex in range(len(ob1)): # for every group after i
                groupStartIndex = N[groupIndex]
                # print("\t\tgroup index",groupIndex)
                # print("group start index", groupStartIndex)
                # print("contents", A[N[groupStartIndex]])
                # print(dimGroup[groupStartIndex])
                groupSize = 10 # HARDCODED VARIBALE, all groups size 10
                # exit()
                scores = np.zeros((groupSize,))
                for rowInGroup in range(groupSize): # for every row in group
                    row = groupStartIndex + rowInGroup
                    # print("\t\t\ton", row)
                    rowContents = A[row]
                    # print("row contents:", rowContents)
                    scores[rowInGroup] = np.dot(currentRow, rowContents) # calculate dot product
                    # print(score)
                # print(scores)

                j = np.argmax(scores) # find max score
                # print("\t\tmax score at:",j)
                if scores[j] > threshold:
                    Y[groupStartIndex + j][p] = 1
                    flag[groupStartIndex + j] = 1

            # print(Y)
            # file = open("file2.txt", "w+")
            # np.set_printoptions(threshold=sys.maxsize)
            # content = str(Y)
            # np.set_printoptions(threshold=None)
            # file.write(content)
            # file.close()
            # print("save Y array")
            # exit()
    # print(Y)

    ## TODO: check Y array with matlab

    return Y

def norm_rows(arr):
    return arr/np.linalg.norm(arr, ord=2, axis=1, keepdims=True)

def normGrayscale(arr):
    min = np.min(arr)
    return (arr - min) / (np.max(arr) - min)
