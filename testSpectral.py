import scipy.io
import numpy as np

from multiObjectMatchingUtil import spectralMatch

mat = scipy.io.loadmat("test_spectral.mat")
# print(mat['vM'])

vM = np.array(mat['vM'])
print(vM.shape)
print(vM)
nFeature = 2
Size = 3

M_out,eigV,timeInfo = spectralMatch(vM,nFeature,Size)

answer = np.identity(10)
