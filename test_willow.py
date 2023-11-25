import sys, os
import numpy as np
import scipy.io
#from dataset.prepareDataset import extractFiles
#from dataset.getDataset import getDataset
from pairwiseMatchingUtil import runGraphMatchBatch
from multiObjectMatchingUtil import runJointMatch
from evaluationUtil import pMatch2perm, evalMMatch
from visualizationUtil import visualizePMatch
from scipy.sparse import csr_matrix
from tqdm import tqdm
# import pdb;  pdb.set_trace()
# ^helpful for debugging, gdb but for python

# all the image classes
classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
showmatch = True
method = 'als'#'pg' # als, pg
numImagesOutput = 5

def main():
    # Load Data
    
    # which image sets to use
    classesToRun = classes 
    #classesToRun = ["Winebottle"]
    #classesToRun = ["Car"] 

    numClasses = len(classesToRun)

    overlaps = np.zeros((numClasses, 2))
    precisions =  np.zeros((numClasses, 2))
    recalls =  np.zeros((numClasses, 2))

    for index, c in enumerate(classesToRun):
        print(f"\nRunning {c}")
        # set variables
        viewList = []
        imgList = []
        datapath = c
        datasetFilePath = os.getcwd() + "/dataset/WILLOW-ObjectClass/"
        files = os.listdir(datasetFilePath + datapath + "/")
        for f in files:
            if f.endswith(".hypercols_kpts.mat"):
                viewList.append(datasetFilePath + datapath + "/" + f)
            elif f.endswith(".png"):
                imgList.append(datasetFilePath + datapath + "/" + f)
        savefile = os.getcwd() + "/results/" + datapath + "/match_kpts.npy" # eventually change to every class run


        # Pairwise matching
        print("Pairwise Matching")
        pMatch = None

        if False:
        # if os.path.exists(savefile): # if pairwise hypercols file exists, load it in
            pMatch = np.load(savefile, allow_pickle=True, fix_imports=True)
        else: # pairwise file does not exist, create it
            # to apply linear matching, set wEdge=0
            # to apply graph mathcing, set wEdge to nonzero number (ex. 1)
            datapath = c
            pMatch = runGraphMatchBatch(datapath,viewList,'all', 0, wEdge=0); # run graph matching
            # print(pMatch.shape)
            np.save(savefile, pMatch, allow_pickle=True, fix_imports=True) # save file

        # import pdb; pdb.set_trace();

        # Construct coordinate matrix C:2*m
        # 2xm matrix
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

        # print(cnt.shape)
        # print(C.shape)
        # print(C)

        # import pdb; pdb.set_trace();



        # Multi-Object Matching
        # methods to try:
        #   'pg': the proposed method, [Multi-Image Semantic Matching by Mining Consistent Features, CVPR 2018]
        #   'spectral': Spectral method, [Solving the multi-way matching problem by permutation synchronization, NIPS 2013]
        #   'matchlift': MatchLift, [Near-optimal joint object matching via convex relaxation, ICML 2014]
        #   'als': MatchALS, [Multi-Image Matching via Fast Alternating Minimization, CVPR 2015]
        print("Multi Object Matching")

        jMatch,jmInfo,tInfo = runJointMatch(pMatch,C,method=method,univsize=10,rank=3,l=1)
        if jMatch is None:
            continue
        # TEMP COMMENT TO DEBUG, rememeber to remove -ere 9/17/2023
        np.save("jMatch", jMatch, allow_pickle=True)
        np.save("jmInfo", jmInfo, allow_pickle=True)

        # print(jMatch)
        # print(jmInfo)
        # print(tInfo)

        # TODO: compare with matlab
        # matlab = scipy.io.loadmat("matlab_outputs.mat")
        # pMatch_mat = np.array(matlab['pMatch'])
        # C_mat = np.array(matlab['C'])
        # jMatch_mat = np.array(matlab['jMatch'])
        # jmInfo_mat = np.array(matlab['jmInfo'])
        # X1_mat = csr_matrix(np.array(matlab['X1']).sum()) # can add .toarray() to convert to numpy
        # X2_mat = csr_matrix(np.array(matlab['X2']).sum())
        # X0_mat = csr_matrix(np.array(matlab['X0']).sum())

        # import pdb; pdb.set_trace();

        # print("Exiting after Multi Object Matching in test_willow")
        # exit()

        jMatch = np.load("jMatch.npy", allow_pickle=True)
        jmInfo = np.load("jmInfo.npy", allow_pickle=True)

        # print("Checking jMatch...")
        # checkMatlabJMatch(jMatch, jMatch_mat)
        # import pdb; pdb.set_trace();

        # Evaluate
        # X1 = pMatch2perm(pMatch); % pairwise matching result
        # import pdb; pdb.set_trace();

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

        overlaps[index] = [o1, o2]
        precisions[index] = [p1, p2]
        recalls[index] = [r1, r2]

        print("Saving images...")
        # TODO: Visualize results -> how?
        if showmatch:
            for i in tqdm(range(pMatch.shape[0])[:numImagesOutput]):
                for j in range(i + 1, pMatch.shape[1])[:numImagesOutput]:
                    if not pMatch[i,j].X is None:
                        visualizePMatch(datapath, pMatch[i,j], method) # mode=3?

        print(f"Finished {c}!")

    print("Overlap Scores (X1, X2):")
    print(overlaps)

    print("Precision Scores (X1, X2):")
    print(precisions)

    print("Recall Scores (X1, X2):")
    print(recalls)

def checkMatlabJMatch(py, matlab):
    tcount, jcount = 0, 0
    for i in range(py.shape[0]):
        for j in range(py.shape[1]):
            if py[i,j] != None:
                tcount += 1
                diff_val = py[i,j].matchInfo+1 - matlab[i,j][0].item()
                if not np.isclose(0, np.linalg.norm(diff_val)):
                    print(f"jMatch {i} {j} differ: ")
                    print(diff_val)
                    print(py[i,j].matchInfo+1)
                    print(matlab[i,j][0].item()[0])
                    jcount += 1
    print(f"Num different: {jcount}/{tcount}")

def checkMatlab(py, matlab):
    tcount, jcount = 0, 0
    for i in range(py.shape[0]):
        for j in range(py.shape[1]):
            if py[i,j] != None:
                tcount += 1
                diff_val = py[i,j] - matlab[i,j]
                if not np.isclose(0, np.linalg.norm(diff_val)):
                    # print(f"jMatch {i} {j} differ: ")
                    # print(diff_val)
                    # print(py[i,j].matchInfo+1)
                    # print(matlab[i,j][0].item()[0])
                    jcount += 1
    print(f"Num different: {jcount}/{tcount}")

if __name__ == '__main__':
    main()
