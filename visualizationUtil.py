import numpy as np
import os
import scipy.io
from classes import pairwiseMatches
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from classes import pairwiseMatches, jointMatchInfo

def visualizePMatch(datapath, pMatch, mode=1):

    if pMatch.X is None:
        return None
    
    linewidth = 1
    markersize = 3
    color = 'y'
    bg = False
    th = 0.1
    direction = 'h'

    view1filename = os.path.join(datapath, pMatch.filename[0])
    view1mat = scipy.io.loadmat(view1filename)
    view1frame = view1mat['frame']

    view2filename = os.path.join(datapath, pMatch.filename[1])
    view2mat = scipy.io.loadmat(view2filename)
    view2frame = view2mat['frame']

    idx = np.where(pMatch.X)[0]

    idx1 = pMatch.matchInfo[idx, 0]
    idx2 = pMatch.matchInfo[idx, 1]

    feat1 = view1frame[:,idx1].astype(np.float64)
    feat2 = view2frame[:,idx2].astype(np.float64)

    # import pdb; pdb.set_trace();

    img1filename = os.path.basename(view1filename).replace(".png.hypercols_kpts.mat", "")
    img2filename = os.path.basename(view2filename).replace(".png.hypercols_kpts.mat", "")

    saveImages(view1mat['img'], view2mat['img'], feat1, feat2, img1filename, img2filename, datapath)

def saveImages(img1, img2, feat1, feat2, f1, f2, datapath):
    num_features = feat1.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, num_features))
    imgs = [img1, img2]
    feats = [feat1, feat2]

    plt.ioff()

    # import pdb; pdb.set_trace();

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=200.0, layout="tight")
    for i in range(2):
        axs[i].imshow(imgs[i])
        for j in range(num_features):
            axs[i].scatter(feats[i][0, j], feats[i][1, j], s= 100, color=(1, 0, 0), marker=f"${j}$")
    fig.suptitle(f"{f1}, {f2} Matches")
    plt.savefig(f"results/WILLOW/{datapath}/{f1}-{f2}.png")
    plt.close(fig)