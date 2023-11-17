import sys, os
import numpy as np
import scipy.io
sys.path.append("../")
from pairwiseMatchingUtil import greedyMatch
from multiObjectMatchingUtil import runJointMatch
from evaluationUtil import pMatch2perm, evalMMatch
from scipy.sparse import csr_matrix
import pickle
import subprocess
from classes import pairwiseMatches
import options, datasets

gen_sim_data = True
TF_ENABLE_ONEDNN_OPTS=0

def test_sim():

    # Generate/Load Simulation

    opts = options.get_opts()
    dname = os.path.join(opts.data_dir,"train")
    if not os.path.exists(opts.data_dir):
        os.makedirs(opts.data_dir)

    simdataset = datasets.get_dataset(opts)


    if len(os.listdir(dname)) == 0 or gen_sim_data:
        print("Generating Synthetic Data Images...")
        simdataset.create_np_dataset(dname, "train")
        print("\tdone.")

    '''
    'convert_dataset'
    create_np_dataset'
    'data_dir'
    'dataset_params'
    'features'
    'gen_adj_mat_noise'
    'gen_init_emb'
    'gen_sample'
    'get_feed_dict'
    'get_placeholders'
    'load_batch'
    'n_pts'
    'n_views'
    'opts'
    'process_features'
    '''

    dataset_images = os.listdir(dname)

    NumViews = simdataset.n_views # 3
    # initial embeddings for optimization
    Nodes = simdataset.features['nodes'] # 30,2
    # edge features
    Edges = simdataset.features['edges'] # 116, 1
    # receiving nodes for edges
    Receivers = simdataset.features['receivers'] # 116,
    # sending nodes for edges
    Senders = simdataset.features['senders'] # 116,
    # ??
    Globals = simdataset.features['globals'] # [0, 0]
    # number of nodes using
    NumNodes = simdataset.n_pts # 30
    # number of edges in this graph
    NumEdges = simdataset.features['n_edge'] # nFeature, 116
    # sparse adjacency matrix of graph
    # num points (p) * num views (n)
    AdjMat = simdataset.features['adj_mat'] # 0: matchInfo, 1: Xraw
    # sparse ground truth adjacency of graph
    TrueAdjMat = simdataset.features['true_adj_mat'] # 0: matchInfo (2, 90),  Xraw (90,)
    # ground truth of matches of graph
    TrueMatch = simdataset.features['true_match'] # (30,)
    


    # TODO
    # datasetfile = "sim_util/synthetic/synthtest/synthtest.pkl"
    # if (not os.path.exists(datasetfile)) or gen_sim_data:
    #     print("Regenerating data...")
    #     d = f"{os.getcwd()}/sim_util/"
    #     subprocess.run(["python", "simulationUtil.py"], cwd=d)
    #     # print("\tcd sim_util; python simulationUtil.py; cd ..")
    #     # exit()
    # with open(datasetfile, 'rb') as fp:
    #     sim_data = pickle.load(fp)
    #     print("Loaded synthetic data.")
    #     # print(sim_data)

    import pdb; pdb.set_trace();

    # import pdb; pdb.set_trace();

    # matchInfo = initSimMatch(AdjMat)
    # MatchInds = np.array(AdjMat[0])
    # Xraw = AdjMat[1]

    # exit()

    # Pairwise matching
    print("Pairwise Matching")
    # matchInfo
    # matchInfo 100,2
    # Xraw, 100

    X = greedyMatch(MatchInds, Xraw)

    # everything in 1 class
    # willow: 40 x 40
    # each element:
    # X = (100,1)
    # Xraw = (100,)
    # matchInfo = (100, 2)
    # nFeature = [10, 10]
    # filename = 2,
    pMatch = pairwiseMatches(MatchInds, NumEdges, Xraw, X)

    import pdb; pdb.set_trace();

    exit()

    # Construct coordinate matrix C:2*m
    # 2xm matrix
    '''
    viewList = list of hypercols_kpts.mat files, 40, images
    C = 2, 400
    d = 2, 10, frame

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

'''
Perform initial matching for simulation data
Inputs:
- X: simlarity scores
Outputs:
- matchInfo: Numpy array of match indices
'''
def initSimMatch(X):
    # X - similairty scores from image i to image j
    # X[0] = 2, 4244
    # X[1] = 4244,

    N = X.shape[0]

    matchInds = np.zeros((N, N, 2))

    for i in range(N):
        print(X[i])
        # TODO: calculate matchInds
        import pdb; pdb.set_trace();
    '''
    X = 100,100

    hypercol files: desc, frame, nfeature, img, filename
    desc1.shape = 640, 10
    viewPair = tuple filenames
    featLocs2 = 2x10, frame (floats)
    kNNInit = 50
    threshScore = 0
    threshRatio = 1

    simScores = 100, (floats)
    matchInds = 100, 2 (indices, ints)
    nMaxInit = 5000
    '''

if __name__ == '__main__':
    test_sim()