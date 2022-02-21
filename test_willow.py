import sys
from dataset.prepareDataset import extractFiles
from dataset.getDataset import getDataset

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

    # construct coordinate matrix C:2*m

    # Multi-Object Matching

    # Evaluate

    # Visualize

if __name__ == '__main__':
    main()
