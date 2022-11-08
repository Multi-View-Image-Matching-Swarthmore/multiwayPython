import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import time
import sys
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.io

from pairwiseMatchingUtil import greedyMatch
from classes import pairwiseMatches, jointMatchInfo

'''
Run Join matching
Inputs:
- pMatch: pairwise matching results, numpy array of pairwiseMatches class
- C: 2*m coordinate matrix, m is the totel number of keypoitns in the image collection
-(optional)method: Joint matching method (spectral, matchLift, als, pg)
-(optional)univsize: how many points per image? (should be 10)
-(optional)rank: not sure
-(optional)l: lambda

Outputs:
- jMatch: Joint matching result, numpy array of pairwiseMatches class
- jmInfo: Joint matching info, numpy array of jointMatchInfo class
- tInfo: Joint matching runtime
'''
def runJointMatch(pMatch, C, method='pg', univsize=10, rank=3, l=1):
    nFeature = np.zeros((pMatch.shape[0], 1))
    # print(nFeature.shape)
    filename = np.empty((nFeature.shape), dtype=str)
    # print(filename.shape)

    nMatches = 0

    # unpack pMatch numpy array
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

    # print(nFeature.shape)


    # initilize variables to run joint matching
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

    #
    for i in range(pDim):
        for j in range(i+1, pDim):
            # try:
            matchList = pMatch[i][j].matchInfo.astype(np.float64) # 100 x 2
            # print(matchList.shape)
            n = int(pMatch[i][j].X.shape[0])
            ind1[0, z:z+n] = matchList[:,0] + cumulativeIndex[i]
            ind2[0, z:z+n] = matchList[:,1] + cumulativeIndex[j]

            flag[0, z:z+n] = pMatch[i][j].X.reshape((1, -1))


            score[0, z:z+n] = normGrayscale(pMatch[i][j].Xraw)
            z += n

    ind1 = ind1.reshape(-1,).astype(int)
    ind2 = ind2.reshape(-1,).astype(int)
    score = score.reshape(-1,)
    flag = flag.reshape(-1,)

    # original scores
    lastIndex = int(cumulativeIndex[cumulativeIndex.shape[0] - 1])

    M = csr_matrix((score, (ind1, ind2)), shape=(lastIndex, lastIndex))

    M = M + M.T


    # binary scores
    Mbin = csr_matrix((flag, (ind1, ind2)), shape=(lastIndex, lastIndex))
    Mbin = Mbin + Mbin.T

    vM = Mbin

    Size = min(univsize, min(nFeature))

    Z = []
    print("Running joint match, problem size = (" + str(vM.shape[0]) + "," + str(Size) + ")")

    method = "als" # debugging line, shoudl eventually remove -ere

    if method == "spectral":
        print("Spectral Matching...")
        M_out, eigV, tInfo = spectralMatch(vM, nFeature, Size)
    elif method == "matchlift":
        print("matchlift (MatchLift) not implemented! Sorry!")
        exit()
    elif method == "als": ## TODO:
        print("MatchALS...")
        M_out, eigV, tInfo, iter = matchALS(vM, nFeature, Size)
        exit()
    elif method == "pg":
        print("pg (proposed method) not implemented! Sorry!")
        exit()
    else:
        print("Unkown Multi-Object Matching method:", method)
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

            # print(rowStart, rowStop)
            # print(colStart, colStop)

            subMatrix = M_out[rowStart:rowStop, colStart:colStop]
            ind1, ind2 = np.nonzero(subMatrix)

            # print(ind1.shape)
            # print(ind1)

            # exit()

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

    # print("end")
    # exit()

    return jMatch,jmInfo,tInfo

def debugNorm(A):
    return np.linalg.norm(A, ord=2)

'''
Run MatchALS Matching
Inputs
- W: numpy matrix of binary scores
- nFeature: numpy array of how many features in each image
- universeSize: same as before (should be 10)
Outputs
- X: sparse binary matrix indicating correspondences
- A: AA^T = X
- runtime: runtime
'''
def matchALS(W, nFeature, universeSize):
    print("In Match ALS method")

    alpha = 20 # was 50
    beta = 0 # 0.1 <-- SP did this
    # maxRank = max(nFeature)*4
    maxRank = universeSize
    pSelect = 1
    tol = 5e-4
    maxIter = 1000
    verbose = False
    eigenvalues = False

    print("Running MatchALS: alpha=%d, beta=%d, maxRank=%d, pSelect=%d" % (alpha, beta, maxRank, pSelect))

    # print(type(W))
    wHeight, wWidth = W.shape

    # w is (400,400)
    # make diagonals zero
    for i in range(min(wHeight, wWidth)):
        W[i,i] = 0.0

    W = (W+W.T)/2.0

    X = W.toarray(order='C') #.astype(np.float32)
    Z = W.toarray(order='C') #.astype(np.float32)
    Y = np.zeros((wHeight, wWidth)) #.astype(np.float32)
    mu = 64

    n = X.shape[0]
    maxRank = min(n, math.ceil(maxRank))
    # A2 = np.random.random((n, maxRank))
    mat = scipy.io.loadmat("A_matrix.mat")
    A = np.array(mat['A'])

    print(A.shape)
    # print(A2.shape)
    # exit()

    nFeature = np.cumsum(nFeature)
    nFeature = np.insert(nFeature, 0, 0)

    print(nFeature)
    print(nFeature.shape)

    # exit()

    start = time.time()

    for i in tqdm(range(maxIter)):

        print("1.", A[:5,:5])

        X0 = X.copy()
        X = Z - (Y.astype(np.float64) - W + beta)/mu # mayber problem here?

        b0 = A.T@A + (alpha/mu) * np.eye(maxRank)
        print("2.", b0[:5, :5])
        b1 = A.T@X # something goes wrong here
        print("3.", b1[:5, :5])
        import pdb; pdb.set_trace()
        B = np.linalg.solve(b0, b1).T
        print("4.", B[:5, :5])

        exit()

        a0 = B.T@B + (alpha/mu) * np.eye(maxRank)
        print("2.", A[:5,:5])
        a1 = B.T@X
        print("3.", A[:5,:5])
        A = np.linalg.solve(a0, a1).T
        print("4.", A[:5,:5])

        X = A*B.T

        print("i: {}, X norm: {:0.5e}, A norm: {:0.5e}, B norm: {:0.5e}".format(i, debugNorm(X), debugNorm(A), debugNorm(B)))
        print("5.", A[:5,:5])
        exit()

        Z = X + Y/mu
        diagZ = np.diagonal(Z)

        # print(diagZ)

        # enforce the self-matching to be null
        for j in range(nFeature.shape[0] - 1):
            ind1 = np.arange(nFeature[j], nFeature[j+1]).astype(int)
            ind1_length = ind1.shape[0]
            Z[ind1][:, ind1] = 0.0
            # print(Z[ind1][:,ind1].shape)
            # exit()
        # Optimize for diaginal elements
        if pSelect == 1:
            for zi in range(Z.shape[0]):
                Z[zi, zi] = 1.0
        else:
            # matlab code:
            # diagZ = proj2kav(diagZ,pSelect*length(diagZ));
            # Z(1:size(Z,1)+1:end) = diagZ;
            print("not implemented proj2kav! Exiting...")
            exit()

        # rounding all elements to [0,1]
        Z = np.clip(Z, 0.0, 1.0)

        Y = Y + mu*(X - Z)

        # import pdb; pdb.set_trace()

        pRes = np.linalg.norm(X.flatten() - Z.flatten())/n
        dRes = mu*np.linalg.norm(X.flatten() - X0.flatten())/n


        if verbose:
            print("Iter = %d, Res = (%e, %e), mu = %f" % (i, pRes, dRes, mu))

        if pRes < tol and dRes < tol:
            break

        if pRes > 10*dRes:
            mu = 2*mu
        elif dRes > 10*pRes:
            mu = mu/2

        if i >= 3:
            exit()

    X = (X+X.T)/2

    end = time.time()

    runtime = end - start
    iterations = i

    print(iterations)

    if iterations >= maxIter - 1:
        print("Algorithm terminated at max iterations. Time = %e, Iter = %d, Res = (%e,%e), mu = %self.fail('message')" % (runtime, iterations, pRes, dRes, mu))
        exit()

    eigenvalues = np.linalg.eig(X)
    ind = np.nonzero(X>0.5)
    X_bin = csr_matrix((X[ind], ind), X.shape)

    print("Algorithm terminated. Time = %e, Iter = %d, Res = (%e,%e), mu = %f" % (runtime, iterations, pRes, dRes, mu))

    return X_bin, eigenvalues, runtime, iterations



'''
NOTE: DO NOT RUN SPECTRAL MATCHING!!! Time wasted here: 10 hours -ere
Run Spectral Matching
Inputs:
- W: numpy matrix of binary scores, Mbin
- nFeature: numpy array of how many features in each image
- universeSize: same as before (should be 10)
Outputs:
- X: numy matrix of new scores, rounded
- V: Eigenvectors
- runtime: runtime
'''
def spectralMatch(W, nFeature, universeSize):
    print("Spectral Matching is broken, exiting...")
    exit()

    k = min(universeSize, W.shape[0])
    start = time.time()
    n = W.shape[0]
    print(n, k, universeSize)
    # print(W.shape)
    # w, V = eigh(W, eigvals=(n-k, n-1))
    w, V = eigs(W, k=k,which="LM") # w = k eigenvalues, V = k eigenvectors
    print(V.shape)
    print(V[0])
    print(w.shape)
    # print(np.real(w))
    import pdb; pdb.set_trace()
    exit()
    ## TODO: what are x and y
    # print()
    # exit(V[:, :k] == V)

    #NOTE: V[:, :k] is the same as V
    V = np.real(V)
    Y = rounding(V, nFeature, threshold=0.5).astype(np.float32)
    # check Y
    mat = scipy.io.loadmat("Y.mat")
    matlabY = np.array(mat['Y'])

    compare = np.isclose(matlabY, Y)

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

'''
NOTE: THIS IS WHY SPECTRAL MATCHING BAD!! (eigenvector ambiguities)
Run rounding on spectral matching
Inputs:

Outputs:

'''
def rounding(A, dimGroup, threshold=0.5): # can we just run k means???
    print("Rounding Function: THIS IS BROKEN!!")
    exit()
    # normalize in order to calculate correlation coefficient of points
    A = norm_rows(A)
    heightA, widthA = A.shape

    mat = scipy.io.loadmat("A.mat")
    matlabA = np.array(mat['A'])

    import pdb; pdb.set_trace()

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
        qwerty = "flag/arr" + str(i) + ".mat"

        mat = scipy.io.loadmat(qwerty)
        matlabFlag = np.array(mat['arr']).flatten()

        pythonFlag = np.nonzero(flag)[0]

        print("i:", i)
        print(matlabFlag.shape, pythonFlag.shape)


        if (len(matlabFlag) != len(pythonFlag)):
            compare = set(matlabFlag).symmetric_difference(pythonFlag)
            import pdb; pdb.set_trace()

        # if i == 18:
        #     import pdb;  pdb.set_trace()
        if flag[i] == 0:
            p += 1 # column we are on
            if p >= Y.shape[1]: # if we need more columns, add a column
                print("Got Bigger")
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
    exit()

    return Y

# helpful method to normalize rows
def norm_rows(arr):
    return arr/np.linalg.norm(arr, ord=2, axis=1, keepdims=True)

# helpful method to normalize array
def normGrayscale(arr):
    min = np.min(arr)
    return (arr - min) / (np.max(arr) - min)
