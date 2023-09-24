import numpy as np

from classes import pairwiseMatches, jointMatchInfo
from scipy.sparse import csr_matrix
import scipy

# debugging function
def debugNorm(A):
    return np.linalg.norm(A, ord=2)


def pMatch2perm(pMatch):
    pHeight = pMatch.shape[0]
    pWidth = pMatch.shape[1] # shld be the same

    nFeature = np.zeros((pHeight, 1))


    # filename = cell(size(pMatch,1),1);
    filename = np.zeros((nFeature.shape[0],), dtype=object)
    emp = 0

    # for i = 1:size(pMatch,1)
    #     for j = i+1:size(pMatch,2)
    for i in range(pHeight):
        for j in range(i+1, pWidth):
            if pMatch[i][j] is not None:
                nFeature[i][0] = pMatch[i][j].nFeature[0]
                nFeature[j][0] = pMatch[i][j].nFeature[1]
                filename[i] = pMatch[i][j].filename[0]
                filename[j] = pMatch[i][j].filename[1]
                # import pdb; pdb.set_trace();
            else:
                emp += 1


    # cumIndex = cumsum([0; nFeature]);
    nFeatureWithZero = np.insert(nFeature, 0, 0)
    cumulativeIndex = np.cumsum(nFeatureWithZero).astype(int)

    lastIndex = int(cumulativeIndex[cumulativeIndex.shape[0] - 1])

    # import pdb; pdb.set_trace();

    # M = sparse(cumIndex(end),cumIndex(end));
    # M = csr_matrix((score, (ind1, ind2)), shape=(lastIndex, lastIndex))

    # M = csr_matrix((lastIndex, lastIndex)) # should be 400
    # could be optimized?
    M = np.zeros((lastIndex, lastIndex))

    for i in range(pHeight):
        for j in range(pWidth):
            if pMatch[i][j] is not None:
                # import pdb; pdb.set_trace();
                matchList = pMatch[i][j].matchInfo.astype(int) # (100,2) or (2,10)
                # correct so it is always (N, 2), bigger dimension first
                if matchList.shape[1] > matchList.shape[0]:
                    matchList = matchList.reshape((matchList.shape[1], -1))
                startrow = cumulativeIndex[i]
                stoprow = cumulativeIndex[i+1]
                startcol = cumulativeIndex[j]
                stopcol = cumulativeIndex[j+1]
                # import pdb; pdb.set_trace();
                M[startrow:stoprow,startcol:stopcol] = csr_matrix((pMatch[i][j].X.reshape((-1,)), (matchList[:, 0], matchList[:, 1])), shape=(int(nFeature[i][0]), int(nFeature[j][0]))).toarray()
                

    # M = csr_matrix((pMatch[i][j].X.reshape((-1,)), (pMatch[i][j].matchInfo[:, 0], pMatch[i][j].matchInfo[:, 1])), shape=(int(nFeature[i]), int(nFeature[j])))
    M = csr_matrix(M)
    M.eliminate_zeros()

    M = M + M.T

    return M

def calcMatch(a, b):
    eps = np.finfo(np.float64).eps
    num = a
    den = b

    # import pdb; pdb.set_trace();

    return (num/den) + eps

def evalMMatch(X,Xgt):
    # X = X > 0;
    # import pdb; pdb.set_trace();
    X = X.toarray()
    X = X > 0
    # Xgt = Xgt > 0;
    Xgt = Xgt.toarray()
    Xgt = Xgt > 0

    # s1 = triu(X,1);
    s1 = np.triu(X, 1)
    # s2 = triu(Xgt,1);
    s2 = np.triu(Xgt, 1)

    # mat2 = scipy.io.loadmat("X2.mat")
    # X_mat = csr_matrix(np.array(mat2['X']).sum()).toarray()
    # import pdb; pdb.set_trace();

    s1nnz = np.count_nonzero(s1)
    s2nnz = np.count_nonzero(s2)
    s1ANDs2 = np.count_nonzero(np.bitwise_and(s1, s2))
    s1ORs2 = np.count_nonzero(np.bitwise_or(s1, s2))

    # overlap = nnz(s1&s2)/(nnz(s1|s2)+eps);
    overlap = calcMatch(s1ANDs2, s1ORs2)
    # precision = nnz(s1&s2)/(nnz(s1)+eps);
    precision = calcMatch(s1ANDs2, s1nnz)
    # recall = nnz(s1&s2)/(nnz(s2)+eps);
    recall = calcMatch(s1ANDs2, s2nnz)

    import pdb; pdb.set_trace();

    return overlap, precision, recall
