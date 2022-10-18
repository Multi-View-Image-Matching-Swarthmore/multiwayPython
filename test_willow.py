import sys, os
import numpy as np
import scipy.io
#from dataset.prepareDataset import extractFiles
#from dataset.getDataset import getDataset
from pairwiseMatchingUtil import runGraphMatchBatch
from multiObjectMatchingUtil import runJointMatch
from evaluationUtil import pMatch2perm, evalMMatch
# import pdb;  pdb.set_trace()
# ^helpful for debugging, gdb but for python

# all the image classes
classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

def main():
    # Load Data
    imageSet = "Car" # default only run with the Car image set
    if len(sys.argv) > 1:
        imageSet = sys.argv[1].capitalize()

    # which image sets to use
    classesToRun = []
    flag = False
    if imageSet == "All":
        classesToRun = classes
    else:
        for i in range(len(classes)):
            if imageSet == classes[i]:
                classesToRun.append(classes[i])
                flag = True
        if flag == False:
            print("Invalid Image Set:", imageSet)
            exit()

    # print(classesToRun)
    # set variables
    viewList = []
    imgList = []
    datapath = classesToRun[0]
    datasetFilePath = os.getcwd() + "/dataset/WILLOW-ObjectClass/"
    files = os.listdir(datasetFilePath + datapath + "/")
    for f in files:
        if f.endswith(".hypercols_kpts.mat"):
            viewList.append(datasetFilePath + datapath + "/" + f)
        elif f.endswith(".png"):
            imgList.append(datasetFilePath + datapath + "/" + f)
    savefile = os.getcwd() + "/results/" + classesToRun[0] + "/match_kpts.npy" # eventually change to every class run


    # Pairwise matching
    print("Pairwise Matching")
    pMatch = None

    if os.path.exists(savefile): # if pairwise hypercols file exists, load it in
        pMatch = np.load(savefile, allow_pickle=True, fix_imports=True)
    else: # pairwise file does not exist, create it
        # to apply linear matching, set wEdge=0
        # to apply graph mathcing, set wEdge to nonzero number (ex. 1)
        datapath = classesToRun[0] # # TODO: edit to run all classes in classesToRun
        pMatch = runGraphMatchBatch(datapath,viewList,'all', 0, wEdge=0); # run graph matching
        # print(pMatch.shape)
        np.save(savefile, pMatch, allow_pickle=True, fix_imports=True) # save file



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



    # Multi-Object Matching
    # methods to try:
    #   'pg': the proposed method, [Multi-Image Semantic Matching by Mining Consistent Features, CVPR 2018]
    #   'spectral': Spectral method, [Solving the multi-way matching problem by permutation synchronization, NIPS 2013]
    #   'matchlift': MatchLift, [Near-optimal joint object matching via convex relaxation, ICML 2014]
    #   'als': MatchALS, [Multi-Image Matching via Fast Alternating Minimization, CVPR 2015]
    print("Multi Object Matching")
    jMatch,jmInfo = runJointMatch(pMatch,C,method='pg',univsize=10,rank=3,l=1)

    print("Exiting after Multi Object Matching in test_willow")
    exit()

# # TODO: all code below this point is unimplemented -ere

    # Evaluate
    # X1 = pMatch2perm(pMatch); % pairwise matching result
    X1 = pMatch2perm(pMatch) # pairwise matching result

    # X2 = pMatch2perm(jMatch); % joint matching result
    X2 = pMatch2perm(jMatch) # joint matching result

    # n_img = length(imgList);
    numImages = len(imgList)

    # n_pts = length(X1)/n_img;
    numPoints = X1.shape[0]/numImages

    # X0 = sparse(repmat(eye(ceil(n_pts)),n_img,n_img)); %groundtruth

    # % evaluate [overlap, precision, recall]
    # [o1,p1,r1] = evalMMatch(X1,X0);
    o1, p1, r1 = evalMMatch(X1, X0)

    # [o2,p2,r2] = evalMMatch(X2,X0);
    o2, p2, r2 = evalMMatch(X2, X0)


    # Visualize
    # if showmatch
    #     %view pairwise matches
    #     for i = 1:size(pMatch,1)
    #         for j = i+1:size(pMatch,2)
    #             clf;
    #             if ~isempty(pMatch(i,j).X)
    #                 subplot('position',[0 0.5 1 0.48]);
    #                 visPMatch(datapath,pMatch(i,j),3,'th',0.01);
    #                 subplot('position',[0 0 1 0.48]);
    #                 visPMatch(datapath,jMatch(i,j),3,'th',0.01);
    #                 fprintf('%d-%d\n',i,j);
    #                 pause
    #             end
    #         end
    #     end
    # end

if __name__ == '__main__':
    main()
