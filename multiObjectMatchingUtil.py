import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import time
import os, sys
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.io

from pairwiseMatchingUtil import greedyMatch
from classes import pairwiseMatches, jointMatchInfo

import ctypes
import pathlib

from munkres import Munkres

'''
Run Joint matching
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
    filename = np.zeros((nFeature.shape[0],), dtype=object)
    # print(filename.shape)

    nMatches = 0

    # import pdb; pdb.set_trace();

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

            # import pdb; pdb.set_trace();

            nMatches += np.sum(pMatch[i][j].X)
            nFeature[i] = pMatch[i][j].nFeature[0]
            nFeature[j] = pMatch[i][j].nFeature[1]
            filename[i] = pMatch[i][j].filename[0]
            filename[j] = pMatch[i][j].filename[1]
            # import pdb; pdb.set_trace();

    # print(nFeature.shape)

    # import pdb; pdb.set_trace();


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
    
    # import pdb; pdb.set_trace();

    M = csr_matrix((score, (ind1, ind2)), shape=(lastIndex, lastIndex)) #start her next time

    # import pdb; pdb.set_trace();

    M = M + M.T


    # binary scores
    # import pdb; pdb.set_trace();
    Mbin = csr_matrix((flag, (ind1, ind2)), shape=(lastIndex, lastIndex))
    Mbin = Mbin + Mbin.T

    vM = Mbin

    Size = min(univsize, min(nFeature))

    Z = []
    print("Running joint match, problem size = (" + str(vM.shape[0]) + "," + str(Size) + ")")

    # import pdb; pdb.set_trace();

    if method == "spectral":
        print("Spectral Matching...")
        M_out, eigV, tInfo = spectralMatch(vM, nFeature, Size)
    elif method == "matchlift":
        print("matchlift (MatchLift) not implemented! Sorry!")
        exit()
    elif method == "als":
        print("MatchALS...")
        #import pdb; pdb.set_trace()
        M_out, eigV, tInfo, iter = matchALS(vM, nFeature, Size)
        if M_out is None:
            return None, None, None
        # exit()
    elif method == "pg":
        print("Proposed Method Matching...")
        M_out, eigV, tInfo, Z = proposedMethod(vM, C, nFeature, Size)
        # print("pg (proposed method) not implemented! Sorry!")
        # print("here !")
        # exit()
    else:
        print("Unkown Multi-Object Matching method:", method)
        exit()

    # import pdb; pdb.set_trace();

    # output/save files?
    jMatch = np.empty((pDim, pDim), dtype=pairwiseMatches)
    # import pdb; pdb.set_trace();

    for i in range(pDim): # check
        for j in range(i+1, pDim):
            # print()
            # [ind1,ind2] = find(   M_out( csum(i)+1:csum(i+1) , csum(j)+1:csum(j+1) )   );
            # return row, col of the nonzero elements indices
            # matlab 1 indexed
            rowStart = cumulativeIndex[i] + 1 - 1
            rowStop = cumulativeIndex[i+1]
            colStart = cumulativeIndex[j] + 1 - 1
            colStop = cumulativeIndex[j+1]

            # print(rowStart, rowStop)
            # print(colStart, colStop)

            subMatrix = M_out[rowStart:rowStop, colStart:colStop].T # transpose, since matlab looks for nonzero elements in column major order first
            ind2, ind1 = np.nonzero(subMatrix) # ??

            # print(ind1.shape)
            # print(ind1)

            # exit()

            # import pdb; pdb.set_trace();

            # if ind1 is not empty aka has nonzero elements
            # if matrix has nonzero elements
            if ind1.shape[0] > 0:
                # remove conflict
                # Xraw = vM(csum(i)+1:csum(i+1),csum(j)+1:csum(j+1));
                Xraw = vM[rowStart:rowStop, colStart:colStop]

                # import pdb; pdb.set_trace();

                # # TODO: help
                # Xraw = Xraw(sub2ind(size(Xraw),ind1,ind2));
                # https://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
                Xraw = np.asarray(Xraw[ind1, ind2])[0]

                # X = greedyMatch([ind1';ind2'],Xraw);
                # def greedyMatch(match, score, nMax=np.inf):
                # import pdb; pdb.set_trace();
                arr = np.vstack((ind1, ind2))
                # import pdb; pdb.set_trace();
                X = greedyMatch(arr, Xraw) # needs a bit more debugging

                # store results
                # jMatch(i,j).matchInfo.match = [ind1';ind2'];
                # jMatch(i,j).X = X;
                # jMatch(i,j).Xraw = Xraw;
                # jMatch(i,j).filename = [filename(i),filename(j)];
                f = np.array([filename[i], filename[j]]) # filename is empty oof

                # jMatch(i,j).nFeature = [nFeature(i),nFeature(j)];
                nf = np.array([nFeature[i], nFeature[j]])

                jMatch[i][j] = pairwiseMatches(arr, nf, Xraw, X)
                jMatch[i][j].filename = f
                # import pdb; pdb.set_trace();

    # jmInfo.eigV = eigV;
    # jmInfo.nFeature = csum;
    # jmInfo.filename = filename;
    # jmInfo.time = timeInfo.time;
    # jmInfo.Z = Z;

    # import pdb; pdb.set_trace();

    
    jmInfo = jointMatchInfo(eigV, cumulativeIndex, filename, tInfo, Z)

    # print("end")
    # exit()

    # import pdb; pdb.set_trace();

    return jMatch,jmInfo,tInfo


'''

'''
def getInds(nFeature, j):
    start, stop = int(nFeature[j]), int(nFeature[j+1])
    return np.arange(nFeature[j], nFeature[j+1]).astype(int)
    

'''
Functions for Proposed Method Matching
- get_newX
    - assignmentoptimalpy (python binding to C code)
- initial_Y
- proj2dpam
    - projR
    - projC
    - proj2pav
    - proj2pavC
'''
'''
assignmentoptimalpy function
Inputs:
- distMatrix (10x10 numpy array)
Outputs:
- assignment (10x1 numpy array)
- cost (int)
'''
def assignmentoptimalpy(distMatrix):
    # import pdb; pdb.set_trace();
    # load the shared library into ctypes
    libname = pathlib.Path.cwd() / "utils" / "assignmentoptimal.so"
    # print(libname)
    c_lib = ctypes.CDLL(libname)

    height, width = distMatrix.shape
    totalSize = height * width
    assignment = np.zeros((height, 1)).flatten()
    cost = 0

    distMatrix2 = distMatrix.flatten()

    # import pdb; pdb.set_trace();

    # TODO STEPHEN
    # void assignmentoptimal(double *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
    c_lib.assignmentoptimalwrapper(assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.addressof(ctypes.c_double(cost)), distMatrix2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(height), int(width))

    # (assignment)
    # print("Made it here!")
    exit()
    return None


'''
get_newX function
Inputs:
- Y: 
- X: 
- C: 
- rho: 
- K: 
- rank: 
- nFeature: 
- var_lambda: 
Outputs:
- X:
'''
def get_newX(Y, X, C, rho, K, rank, nFeature, var_lambda):
    # print("getting new X...")
    # geometric constraint
    n = len(nFeature) - 1
    M = np.zeros((2*n, X.shape[1]))
    for i in range(n):
        ind1 = getInds(nFeature, i)
        M[2*i:2*i+2,:] = C[:,ind1]@X[ind1,:]
    # update Z
    # import pdb; pdb.set_trace()
    U, S, V = np.linalg.svd(M, full_matrices=False) # econ setting (dont worry about it)
    # import pdb; pdb.set_trace()
    S = np.diag(S)
    Z = U[:,:S.shape[0]]@S[:,:rank]@V[:rank,:] 

    munkres_algo = Munkres()

    # update X
    for i in range(n):
        # print(i)
        ind1 = getInds(nFeature, i)
        ind1_length = ind1.shape[0]
        Ci = C[:,ind1]
        Zi = Z[2*i:2*i+2, :]
        Yi = Y[ind1,:]
        # hungarian
        D = var_lambda*cdist(Ci.T,Zi.T, "euclidean") # still issues
        distMatrix = D - rho*Yi - np.min(D - rho*Yi)
        # assignment = assignmentoptimalpy(distMatrix.astype(np.float64)) # todo implement assignmentoptimal
        # just import it from Matlab
        # aofilename = "ao" + str(i+1) + ".mat"
        # aoFilenameFull = pathlib.Path.cwd() / "assignmentoptimals" / aofilename
        # mat = scipy.io.loadmat(aoFilenameFull)
        # assignment = np.array(mat['assignment']).reshape(1,-1)[0]

        # import pdb; pdb.set_trace()
        assignment = np.array(munkres_algo.compute(distMatrix.astype(np.float64)))[:, 1]

        Xhi = np.zeros((ind1_length, K))
        # print(Xhi.shape)
        q = np.where(assignment >= 1)[0]
        Xhi[q, assignment[q]-1] = 1 # Matlab is 1 indexed
        X[ind1,:] = Xhi
    return X, M
'''
initial_Y function
Inputs:

Outputs: 
- Y
'''
def initial_Y(Y0, W, nFeature):
    # print("initialize Y...")
    # import pdb; pdb.set_trace();
    tol = 1e-4
    maxIter = 500
    lambdas = [1e0, 1e1, 1e2, 1e3, 1e4]
    Y = Y0
    m = W.shape[0]
    start = time.time()
    # import pdb; pdb.set_trace();
    Y_YYtd = np.zeros(Y0.shape)
    for i in range(len(lambdas)):
        for iter in range(maxIter):
            Y0 = Y
            # compute gradient and stepsize
            normr = 0
            for j in range(nFeature.shape[0] - 1):
                ind1 = getInds(nFeature, j)
                # import pdb; pdb.set_trace();
                Yi = Y[ind1,:]
                YitYi = Yi.T@Yi
                Y_YYtd[ind1,:] = Yi@YitYi
                # step size of regularizer
                Hr = lambdas[i]*(3*np.linalg.norm(Y[ind1,:].T@Y[ind1,:],2) + 1) # future note: scipy for sparse matrix might be better
                normr = max(Hr, normr) # normalized regularizer?

            gi = lambdas[i]*(Y_YYtd - Y)
            gij = -W@Y + Y@(Y.T@Y)
            g = gi + gij # gradient
            # import pdb; pdb.set_trace();
            # W norms different? -ere 3/19/23
            st = 3*np.linalg.norm(Y.T@Y,2) + scipy.sparse.linalg.norm(W,1) + normr #stepsize
            # import pdb; pdb.set_trace();

            # update and project
            Y = Y - g/st # oh shoot we need more variables in method ack
            for k in range(nFeature.shape[0] - 1):
                ind2 = getInds(nFeature, k)
                Y[ind2,:] = proj2dpam(Y[ind2,:], 1e-2)

            # import pdb; pdb.set_trace();
            
            RelChg = np.linalg.norm(Y - Y0)/math.sqrt(m)
            
            # print("lambda = %d,  iter = %d, Res = %.6f" % (lambdas[i], iter, RelChg))

            if RelChg < tol:
                break
    return Y

'''
projR function (for proj2dpam)
Inputs:

Outputs:
'''
def projR(X):
    X_return = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        X_return[i,:] = proj2pav(X[i,:].T).T
    return X_return 

'''
projC function (for proj2dpam)
Inputs:

Outputs:
'''
def projC(X):
    X_return = np.zeros((X.shape[0], X.shape[1]))
    for j in range(X.shape[1]):
        X_return[:,j] = proj2pavC(X[:,j])
    return X_return 

'''
proj2pav function (for proj2dpam)
Inputs:

Outputs:
'''
def proj2pav(y):
    x = np.zeros((y.shape[0],)) # intialize x

    y = np.clip(y, 0, None)
    
    if np.sum(y) < 1:
        x = y
    else:
        u = np.flip(np.sort(y))
        sv = np.cumsum(u)
        rho_divide = np.divide((sv - 1), np.arange(1, u.shape[0] + 1))
        rho = np.argwhere(u > rho_divide)[-1][0] # index is one less than matlab btw
        theta = max(0, (sv[rho] - 1) / (rho + 1)) # needs +1 due to matlab indexing
        x = np.maximum(y-theta,0) # elementwise max
        # import pdb; pdb.set_trace();
    
    return x

'''
proj2pavC function (for proj2dpam)
Inputs:

Outputs:

Info:
% project an n-dim vector y to the simplex Dn
% Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}

% (c) Xiaojing Ye
% xyex19@gmail.com
%
% Algorithm is explained as in the linked document
% http://arxiv.org/abs/1101.6081
% or
% http://ufdc.ufl.edu/IR00000353/
%
% Jan. 14, 2011.
'''
def proj2pavC(y):

    m = len(y)
    bget = False

    s = np.flip(np.sort(y))
    tmpsum = 0

    for ii in range(m-1):
        tmpsum += s[ii]
        tmax = (tmpsum - 1)/ (ii + 1) # matlab index + 1 offset
        if tmax >= s[ii + 1]:
            # import pdb; pdb.set_trace();
            bget = True
            break
    
    if not bget:
        tmax = (tmpsum + s[m - 1] - 1)/m # index offset

    x = np.maximum(y-tmax, 0)

    # import pdb; pdb.set_trace();
    
    return x

'''
proj2dpam function
Inputs:
- Y:
- tol: 
Outputs:
- X: 
'''
def proj2dpam(Y,tol): # todo
    X0 = Y
    X = Y
    I2 = 0

    for i in range(10):
        # row projection
        X1 = projR(X0 + I2)
        I1 = X1 - (X0 + I2)
        X2 = projC(X0 + I1)
        I2 = X2 - (X0 + I1)

        numElements = X.shape[0] * X.shape[1]
        chg = np.sum(abs(X2 - X))/numElements

        X = X2
        if chg < tol:
            # import pdb; pdb.set_trace();
            return X
    
    # should hopefully never get here -ere
    print("Error in proj2dpam! Did not converge.")
    exit()
    

'''
Run the Proposed Method Matching
Inputs:
- W: numpy matrix of binary scores, m*m
- C: numpy coordinate matric, 2*m
- nFeature: numpy array of how many features in each image
- k: number of selected features
Outputs:
- XP: joint matching result, m*m
- X: mapping from image features to selected feature space, m*k
- M: coordinates of selected features
'''
def proposedMethod(vM, C, nFeature, Size, var_lambda=1, rank=3, verbose=False):
    # print("In Proposed Method function.")
    # print("Inputs: nFeature=", nFeature[0])

    tol_Y = 1e-4
    var_lambda = 1 # weight of geometric constant, redundant, made as param ????
    maxIter = 500
    maxIter_Y = 500
    k = Size
    # import pdb; pdb.set_trace();
    W = vM

    print("Running Proposed Method: lambda=%d, rank=%d, tol_Y=%d, maxIter=%d, maxIter_Y=%d" % (var_lambda, rank, tol_Y, maxIter, maxIter_Y))

    # import pdb; pdb.set_trace()

    # print(vM)
    m = vM.shape[0]
    
    nFeature = np.cumsum(nFeature)
    nFeature = np.insert(nFeature, 0, 0)

    #Y = np.random.rand(m, k) # initialize Y randomly, doesn't matter, random guess
    # using matlab saved Y as file
    # import pdb; pdb.set_trace()
    if m == 400:
        mat = scipy.io.loadmat("Y400.mat")
    elif m == 500:
        mat = scipy.io.loadmat("Y500.mat")
    elif m == 1088:
        mat = scipy.io.loadmat("Y1088.mat")
    elif m == 660:
        mat = scipy.io.loadmat("Y660.mat")
    elif m == 1080: # weird
        mat = scipy.io.loadmat("Y1080.mat")
    else:
        if not os.path.exists("Y.mat"):
            Y = np.random.rand(m, k)
            scipy.io.savemat("Y.mat", {'Y': Y})
        mat = scipy.io.loadmat("Y.mat")
    Y = np.array(mat['Y'])

    start = time.time()

    Y = initial_Y(Y, W, nFeature)

    # if os.path.exists("Y.npy"):
    #     Y = np.load("Y.npy")
    # else:
    #     Y = initial_Y(Y, W, nFeature) # initialize Y by projected gradient descent
    #     np.save("Y.npy", Y)

    # print(Y.shape)
    # import pdb; pdb.set_trace()

    # initialize X
    U = np.zeros((m,k))
    # C / ((std(C(1,:)) + std(C(2,:)))/2)
    C_norm_factor = (np.std(C[0,:]) + np.std(C[1,:])) / 2.0
    C_norm = C / C_norm_factor
    # import pdb; pdb.set_trace()

    X,_ = get_newX(U, Y, C_norm, 0, k, rank, nFeature, var_lambda)
    # speed up get new X
    # if os.path.exists("X_init.npy"):
    #     X = np.load("X_init.npy")
    # else:
    #     X,_ = get_newX(U, Y, C_norm, 0, k, rank, nFeature, var_lambda)
    #     np.save("X_init.npy", X)

    # update Y X Z
    Rho = [1e0, 1e1, 1e2]
    Iter = 0
    # import pdb; pdb.set_trace()
    for i in range(len(Rho)):
        rho = Rho[i]
        for iter1 in range(1, maxIter+1):
            # print(iter1)
            # update Y
            for iter_Y in range(maxIter_Y):
                t1_start = time.time()
                Y0 = Y
                # import pdb; pdb.set_trace()
                g = - W@Y + Y@(Y.T @ Y) + rho*(Y - X)
                # import pdb; pdb.set_trace()
                st = 3*np.linalg.norm(Y.T@Y) + scipy.sparse.linalg.norm(W) + rho
                Y = Y - g/st
                for i2 in range(nFeature.shape[0] - 1):
                    ind1 = getInds(nFeature, i2)
                    # import pdb; pdb.set_trace()
                    Y[ind1, :] = proj2dpam(Y[ind1,:], 1e-2)
                t1_stop = time.time()
                t1 = t1_stop - t1_start
                RelChg = np.linalg.norm(Y - Y0)/math.sqrt(m)

                if verbose:
                    print("Iter = %d, iter_Y  = %d, Res = %e, t = %f\n" % (iter1, iter_Y, RelChg, t1))

                if RelChg < tol_Y:
                    # import pdb; pdb.set_trace()
                    break
            # update X
            X0 = X
            X, M = get_newX(Y, X, C_norm, rho, k, rank, nFeature, var_lambda)
           
            if np.sum(abs(X - X0)) < 1: # check?
                break

        Iter = Iter + iter1 # wtf come back to this

    # import pdb; pdb.set_trace()

    # overall match
    stop = time.time()
    runtime = stop - start
    iterations = Iter
    XP = X@X.T
    if verbose:
        print("Time = %fs, Overall Iter = %d" % (runtime, iterations))

    # import pdb; pdb.set_trace()

    # M_out, eigV, tInfo, Z
    return XP, X, runtime, M

# debugging function
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
def matchALS(W, nFeature, universeSize, verbose=False):
    # print("Running Match ALS...")

    alpha = 20 # was 50
    beta = 0 # 0.1 <-- SP did this
    # maxRank = max(nFeature)*4
    maxRank = universeSize
    pSelect = 1
    tol = 5e-4
    maxIter = 1000
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
    if n == 400:
        mat = scipy.io.loadmat("als_A400.mat")
    elif n == 500:
        mat = scipy.io.loadmat("als_A500.mat")
    elif n == 1088:
        mat = scipy.io.loadmat("als_A1088.mat")
    elif n == 660:
        mat = scipy.io.loadmat("als_A660.mat")
    elif n == 1080: # weird
        mat = scipy.io.loadmat("als_A1080.mat")
    else:
        if not os.path.exists("als_A.mat"):
            A = np.random.rand((n, maxRank))
            scipy.io.savemat("als_A.mat", {'A': A})
        mat = scipy.io.loadmat("als_A.mat")
    A = np.array(mat['A'])

    # print(A.shape)

    nFeature = np.cumsum(nFeature)
    nFeature = np.insert(nFeature, 0, 0)

    # print(nFeature)
    # print(nFeature.shape)

    start = time.time()

    # for i in tqdm(range(maxIter)):
    for i in range(maxIter):

        X0 = X.copy()
        X = Z - (Y.astype(np.float64) - W + beta)/mu

        b0 = A.T@A + (alpha/mu) * np.eye(maxRank)
        #import pdb; pdb.set_trace()
        b1 = A.T@X 
        B = np.linalg.solve(b0, b1).T

        a0 = B.T@B + (alpha/mu) * np.eye(maxRank)
        a1 = B.T@X.T
        A = np.linalg.solve(a0, a1).T

        X = A@B.T

        # print("i: {}, X norm: {:0.5e}, A norm: {:0.5e}, B norm: {:0.5e}".format(i, debugNorm(X), debugNorm(A), debugNorm(B)))


        Z = X + Y/mu
        diagZ = np.diagonal(Z)

        # enforce the self-matching to be null
        for j in range(nFeature.shape[0] - 1):
            start, stop = int(nFeature[j]), int(nFeature[j+1])
            ind1 = np.arange(nFeature[j], nFeature[j+1]).astype(int)
            ind1_length = ind1.shape[0]
            Z[start:stop, start:stop] = 0.0
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

        Z = np.clip(Z, 0.0, 1.0)

        Y = Y + mu*(X - Z)

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

    X = (X+X.T)/2

    end = time.time()

    runtime = end - start
    iterations = i

    # print(iterations)

    if iterations >= maxIter - 1:
        print("Algorithm terminated at max iterations. Time = %e, Iter = %d, Res = (%e,%e), mu = %self.fail('message')" % (runtime, iterations, pRes, dRes, mu))
        #exit()
        return None, None, None, None

    eigenvalues = np.linalg.eig(X)
    ind = np.nonzero(X>0.5)
    temp = np.array(X[ind]).flatten()
    X_bin = csr_matrix((temp, ind), X.shape)

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
    X = np.matmul(Y,Y.T)
    end = time.time()

    runtime = end - start
    print(runtime)

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

    flag = np.zeros((heightA,1)) # indicates in row already has been assigned a max
    Y = np.zeros((heightA, widthA)) # ends up 400 x 16?
    p = -1
    scores = np.zeros((10))

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
            ob1 = np.where(N > i)[0][:-1] # indices of all the groups after i
            currentRow = A[i]
            for groupIndex in range(len(ob1)): # for every group after i
                groupStartIndex = N[groupIndex]
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
