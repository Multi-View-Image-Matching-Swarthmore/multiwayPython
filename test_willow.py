import sys
from dataset.prepareDataset import extractFiles
from dataset.getDataset import getDataset
from pairwiseMatchingUtil import runGraphMatchBatch

classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

def main():
    # print("hello world")

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

    getDataset(classesToRun)

    # Pairwise matching
    # if pairwise matching file exists, load it in
    # exists at result/willow/imageSet/match_kpts.npy

    #i if not, create it
    # to apply linear matching, set 'wEdge' as 0;
    # to apply graph matching, set wEdge as an nonzero number, e.g., 1
    # pMatch = runGraphMatchBatch(numpy files,hypercols_kpts,'all',[],'wEdge', 0);
    # save(savefile,'pMatch');


    # construct coordinate matrix C:2*m

    # Multi-Object Matching

    # Evaluate

    # Visualize

if __name__ == '__main__':
    main()
