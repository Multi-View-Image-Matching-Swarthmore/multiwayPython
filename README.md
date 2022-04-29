# Multi-Object Matching (for Python)
Erebus Oh, Professor Stephen Phillips
<eoh2@swarthmore.edu>

This repository is a Python implementation of [Multi-Object Matching](https://github.com/zju-3dv/multiway). To view their README, see Multi-ObjectMatchingREADME.md.

# Setup

## Anaconda Environment
Set up and activate the Anaconda Environment by running the command:
```
conda env create --file environment.yml
conda activate multiway
```

## Dataset
We use the WILLOW-ObjectClass Dataset and Alexnet. Follow the steps below:

1. Download [WILLOW-ObjectClass Dataset](http://www.di.ens.fr/willow/research/graphlearning/) at ```dataset/```

```
cd dataset/
wget http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip
unzip WILLOW-ObjectClass_dataset.zip
# remove problematic image and annotation
rm -f WILLOW-ObjectClass/Face/image_0160.*
# there is an annotation error in Cars_030a.mat (coordinate swap between 6th and 7th keypoint), replace it with the correct one
mv Cars_030a.mat WILLOW-ObjectClass/Car/
```
2. Download [Alexnet Weights](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) at ```hypercols/```, and then extract feature descriptor ***hypercolumn*** from AlexNet.
```
cd ../hypercols/
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
python run_willow.py
```

# Running the code
To run the code, type the command:
```
python test_willow
```
