import sys, os
import numpy as np
import scipy.io
from dataset.prepareDataset import extractFiles
from dataset.getDataset import getDataset
from pairwiseMatchingUtil import runGraphMatchBatch
# import pdb;  pdb.set_trace()

classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

def main():
    # print("hello world")
    imageSet = "Car"
    if len(sys.argv) > 1:
        imageSet = sys.argv[1].capitalize()
    # print(imageSet) # which image set to use?

    # Load data
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

    print(classesToRun)
    viewList = []
    datapath = classesToRun[0]
    datasetFilePath = os.getcwd() + "/dataset/WILLOW-ObjectClass/"
    files = os.listdir(datasetFilePath + datapath + "/")
    for f in files:
        if f.endswith(".hypercols_kpts.mat"):
            viewList.append(datasetFilePath + datapath + "/" + f)
    savefile = os.getcwd() + "/results/" + classesToRun[0] + "/match_kpts.npz" # eventually change to every class run

    # Pairwise matching
    # if pairwise matching file exists, load it in
    pMatch = None

    if os.path.exists(savefile):
        pMatch = np.load(savefile)
    else: # calculate matches
        datapath = classesToRun[0]
        # print(savefile)
        pMatch = runGraphMatchBatch(datapath,viewList,'all','wEdge');
        np.savez(savefile, pMatch)

    print(pMatch)

    # construct coordinate matrix C:2*m
    # 2xm matrix

    # for i = 1:length(viewList)
    # views(i) = load(sprintf('%s/%s',datapath,viewList{i}));
    # cnt(:,i) = sum(views(i).frame,2)/double(views(i).nfeature);
    # C = [C,views(i).frame - repmat(cnt(:,i),1,views(i).nfeature)];
    # concates horizontally
    # C and views i - repmat

    # repmat(cnt(:,i),     1,    views(i).nfeature)

    C = None
    cNotSet = True # very scuffed way of adding numpy array
    cnt = np.zeros((2, len(viewList)))
    # print(cnt.shape)
    for i in range(len(viewList)):
        # print(viewList[i])
        # load hypercols file
        viewMat = scipy.io.loadmat(viewList[i])
        viewFrame = viewMat['frame']
        viewNFeature = viewMat['nfeature'].astype(np.float64)[0][0]
        # print(viewNFeature)
        cntTemp = np.sum(viewFrame, axis=1)/viewNFeature
        cnt[:,i] = cntTemp
        # print(cnt[:, i].reshape((-1,1)))
        # print(np.tile(cnt[:,i].reshape((-1,1)), (1, int(viewNFeature))))
        # t = np.tile(cnt[:,i], (0, int(viewNFeature))).reshape(2,10)
        d = viewFrame - np.tile(cnt[:,i].reshape((-1,1)), (1, int(viewNFeature)))
        # print(d.shape)
        # print(C)
        if cNotSet: # scuffed adding of numpy arrays
            C = d
            cNotSet = False
        else:
            C = np.concatenate((C,d),axis=1)

    # print(cnt.shape)
    # print(C.shape)
    # print(C)
    exit()

    # Multi-Object Matching

    # Evaluate

    # Visualize

if __name__ == '__main__':
    main()
