import sys, os
import numpy as np
import scipy.io
from pairwiseMatchingUtil import greedyMatch
from multiObjectMatchingUtil import runJointMatch
from evaluationUtil import pMatch2perm, evalMMatch
from scipy.sparse import csr_matrix
import pickle
# from sim_util import options

def test_sim():

    # Generate/Load Simulation
    # TODO
    datasetfile = "sim_util/synthetic/synthtest/synthtest.pkl"
    with open(datasetfile, 'rb') as fp:
        sim_data = pickle.load(fp)
        print("Loaded synthetic data.")
        # print(sim_data)
    print(sim_data['adj_mat'])

    exit()

    # Pairwise matching
    print("Pairwise Matching")
    # matchInfo
    # Xraw = AdjMat

    X = greedyMatch(matchInfo, Xraw)

    pMatch = None

    ## get pMatch from synth_dataset, gen_sample()

    import pdb; pdb.set_trace();

    # Construct coordinate matrix C:2*m
    # 2xm matrix
    '''
    print("Constructing Coordinate Matrix")
    C = None
    cNotSet = True # very scuffed way of adding numpy array
    cnt = np.zeros((2, len(viewList)))
    # print(cnt.shape)
    for i in range(len(viewList)): # for every image
        viewMat = scipy.io.loadmat(viewList[i])
        viewFrame = viewMat['frame']
        viewNFeature = viewMat['nfeature'].astype(np.float64)[0][0]
        cntTemp = np.sum(viewFrame, axis=1)/viewNFeature
        cnt[:,i] = cntTemp
        d = viewFrame - np.tile(cnt[:,i].reshape((-1,1)), (1, int(viewNFeature)))
        if cNotSet: # scuffed adding of numpy arrays
            C = d
            cNotSet = False
        else:
            C = np.concatenate((C,d),axis=1)

    '''



    # Multi-Object Matching
    # methods to try:
    #   'pg': the proposed method, [Multi-Image Semantic Matching by Mining Consistent Features, CVPR 2018]
    #   'spectral': Spectral method, [Solving the multi-way matching problem by permutation synchronization, NIPS 2013]
    #   'matchlift': MatchLift, [Near-optimal joint object matching via convex relaxation, ICML 2014]
    #   'als': MatchALS, [Multi-Image Matching via Fast Alternating Minimization, CVPR 2015]
    print("Multi Object Matching")

    '''

    jMatch,jmInfo,tInfo = runJointMatch(pMatch,C,method=method,univsize=10,rank=3,l=1)
    # TEMP COMMENT TO DEBUG, rememeber to remove -ere 9/17/2023
    np.save("jMatch", jMatch, allow_pickle=True)
    np.save("jmInfo", jmInfo, allow_pickle=True)

    # print(jMatch)
    # print(jmInfo)
    # print(tInfo)

    # TODO: compare with matlab
    matlab = scipy.io.loadmat("matlab_outputs.mat")
    pMatch_mat = np.array(matlab['pMatch'])
    C_mat = np.array(matlab['C'])
    jMatch_mat = np.array(matlab['jMatch'])
    jmInfo_mat = np.array(matlab['jmInfo'])
    X1_mat = csr_matrix(np.array(matlab['X1']).sum()) # can add .toarray() to convert to numpy
    X2_mat = csr_matrix(np.array(matlab['X2']).sum())
    X0_mat = csr_matrix(np.array(matlab['X0']).sum())

    # import pdb; pdb.set_trace();

    # print("Exiting after Multi Object Matching in test_willow")
    # exit()

    jMatch = np.load("jMatch.npy", allow_pickle=True)
    jmInfo = np.load("jmInfo.npy", allow_pickle=True)

    # print("Checking jMatch...")
    # checkMatlabJMatch(jMatch, jMatch_mat)
    # import pdb; pdb.set_trace();

    '''

    # Evaluate
    # X1 = pMatch2perm(pMatch); % pairwise matching result
    # import pdb; pdb.set_trace();

    '''

    X1 = pMatch2perm(pMatch) # pairwise matching result
    # print("Checking X1...")
    # checkMatlab(X1, X1_mat)
    # import pdb; pdb.set_trace();

    # X2 = pMatch2perm(jMatch); % joint matching result
    X2 = pMatch2perm(jMatch) # joint matching result
    # print("Checking X2...")
    # checkMatlab(X2, X2_mat)
    # import pdb; pdb.set_trace();

    

    # mat1 = scipy.io.loadmat("X1.mat")
    # X1_mat = csr_matrix(np.array(mat1['X1']).sum()).toarray()
    # mat2 = scipy.io.loadmat("X2.mat")
    # X2_mat = csr_matrix(np.array(mat2['X2']).sum()).toarray()

    # import pdb; pdb.set_trace();

    # n_img = length(imgList);
    numImages = len(imgList)
    # n_pts = length(X1)/n_img;
    numPoints = X1.shape[0]/numImages

    # X0 = sparse(repmat(eye(ceil(n_pts)),n_img,n_img)); %groundtruth
    X0 = csr_matrix(np.tile(np.eye(int(np.ceil(numPoints))), (numImages, numImages)))
    # mat1 = scipy.io.loadmat("X0.mat")
    # X1_mat = csr_matrix(np.array(mat1['X0']).sum())

    # import pdb; pdb.set_trace();s


    # % evaluate [overlap, precision, recall]
    # [o1,p1,r1] = evalMMatch(X1,X0);
    o1, p1, r1 = evalMMatch(X1, X0)

    # [o2,p2,r2] = evalMMatch(X2,X0);
    o2, p2, r2 = evalMMatch(X2, X0)

    # import pdb; pdb.set_trace();

    print("Overlap, Precision Recall")
    print(f"X1: {o1}, {p1}, {r1}")
    print(f"X2: {o2}, {p2}, {r2}")

    '''

    print("Finished!")

if __name__ == '__main__':
    test_sim()
    # print("Hello")