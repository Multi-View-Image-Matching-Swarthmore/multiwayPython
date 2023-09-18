import numpy as np

from classes import pairwiseMatches, jointMatchInfo
from scipy.sparse import csr_matrix

# debugging function
def debugNorm(A):
    return np.linalg.norm(A, ord=2)


def pMatch2perm(pMatch):
    pHeight = pMatch.shape[0]
    pWidth = pMatch.shape[1] # shld be the same

    nFeature = np.zeros((pHeight, 1))

    # import pdb; pdb.set_trace();


    # filename = cell(size(pMatch,1),1);
    filename = np.zeros(nFeature.shape, dtype=str)

    # for i = 1:size(pMatch,1)
    #     for j = i+1:size(pMatch,2)
    for i in range(pHeight):
        for j in range(i+1, pWidth):
            # if ~isempty(pMatch(i,j).nFeature)
    #             nFeature(i) = pMatch(i,j).nFeature(1);
    #             nFeature(j) = pMatch(i,j).nFeature(2);
    #         end
    #         if ~isempty(pMatch(i,j).filename)
    #             filename(i) = pMatch(i,j).filename(1);
    #             filename(j) = pMatch(i,j).filename(2);
            if pMatch[i][j].nFeature == 0:
                print("nFeature is empty for:", i, j)
                exit()
            if len(pMatch[i][j].filename) == 0:
                print("filename is empty for:", i, j)
                exit()

            nFeature[i] = pMatch[i][j].nFeature
            nFeature[j] = pMatch[i][j].nFeature # check
            filename[i] = pMatch[i][j].filename[0]
            filename[j] = pMatch[i][j].filename[1]


    # cumIndex = cumsum([0; nFeature]);
    nFeatureWithZero = np.insert(nFeature, 0, 0)
    cumulativeIndex = np.cumsum(nFeatureWithZero).astype(int)

    lastIndex = int(cumulativeIndex[cumulativeIndex.shape[0] - 1])

    # import pdb; pdb.set_trace();

    # M = sparse(cumIndex(end),cumIndex(end));
    # M = csr_matrix((score, (ind1, ind2)), shape=(lastIndex, lastIndex))

    # M = csr_matrix((lastIndex, lastIndex)) # should be 400
    data = np.empty([])
    indices = cumulativeIndex
    indptr = np.array([])

    for i in range(pHeight):
        for j in range(pWidth):
            if pMatch[i][j] is not None:
                data = np.append(data, pMatch[i][j].X.reshape((-1,)))
                # rows = np.append(rows, pMatch[i][j].matchInfo[:, 0])
                # cols = np.append(cols, pMatch[i][j].matchInfo[:, 1])
                # indices = np.append(cumulativeIndex[i], cumulativeIndex[j])
                indptr = np.append(indptr, pMatch[i][j].matchInfo)
                import pdb; pdb.set_trace();


    
    import pdb; pdb.set_trace();

    # M = csr_matrix((pMatch[i][j].X.reshape((-1,)), (pMatch[i][j].matchInfo[:, 0], pMatch[i][j].matchInfo[:, 1])), shape=(int(nFeature[i]), int(nFeature[j])))
    M = csr_matrix((data, indices, indptr), shape=((lastIndex,lastIndex)))
    M.eliminate_zeros()

    # for i = 1:size(pMatch,1)
    #     for j = i+1:size(pMatch,2)
    #         if ~isempty(pMatch(i,j).matchInfo)
    #             matchList = double(pMatch(i,j).matchInfo.match);
    #             M(cumIndex(i)+1:cumIndex(i+1),cumIndex(j)+1:cumIndex(j+1)) = ...
    #                 sparse(matchList(1,:),matchList(2,:),pMatch(i,j).X,nFeature(i),nFeature(j));
    #         end
    #     end
    # end
    # M = M + M';

    M = M + M.T

    return M

def evalMMatch(X,Xgt):
    overlap, precision, recall = None, None, None
    # X = X > 0;
    # Xgt = Xgt > 0;

    # s1 = triu(X,1);
    # upper triangle matrix: https://www.mathworks.com/help/matlab/ref/triu.html
    # s2 = triu(Xgt,1);

    # is there a built in library for this?
    # overlap = nnz(s1&s2)/(nnz(s1|s2)+eps);
    # precision = nnz(s1&s2)/(nnz(s1)+eps);
    # recall = nnz(s1&s2)/(nnz(s2)+eps);
    # nnz - number of nonzero matrix elements: https://www.mathworks.com/help/matlab/ref/nnz.html

    return overlap, precision, recall
