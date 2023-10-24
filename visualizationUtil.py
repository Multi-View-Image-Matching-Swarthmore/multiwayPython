import numpy as np
import os
import scipy.io
from classes import pairwiseMatches
import tqdm

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

    import pdb; pdb.set_trace();

    return None
    
'''
            
view1 = load([datapath,'/',pMatch.filename{1}],'img','frame','filename');
'dataset/WILLOW-ObjectClass/Car//Cars_000a.png.hypercols_kpts.mat'
img = image, uint8
frame (2x10)
filename png

[imgInput,margin] = appendimages(view1.img,view2.img,direction,10);
% imgInput = mat2gray(imgInput);
imshow(imgInput); hold on;
iptsetpref('ImshowBorder','tight');

feat2v = feat2;
switch direction
    case 'h'
        feat2v(1,:) = feat2v(1,:) + size(view1.img,2) + margin;
    case 'v'
        feat2v(2,:) = feat2v(2,:) + size(view1.img,1) + margin;
end

switch mode
    case 1 % lines
        for i = 1:size(feat1,2)
            if bg
            plot([ feat1(1,i), feat2v(1,i) ]...
                ,[ feat1(2,i), feat2v(2,i) ],...
                '-o','LineWidth',linewidth+1,'MarkerSize',markersize+1,...
                'color', 'k', 'MarkerFaceColor', 'k');
            end
            plot([ feat1(1,i), feat2v(1,i) ]...
                ,[ feat1(2,i), feat2v(2,i) ],...
                '-o','LineWidth',linewidth,'MarkerSize',markersize,...
                'color',color{mod(i,length(color))+1},'MarkerFaceColor',color{mod(i,length(color))+1});
        end
        
    case 2 % color dots
%         c = linspace(0,1,size(feat1,2));
        c = mat2gray(sum(feat1,1)); 
        scatter(feat1(1,:),feat1(2,:),markersize,c,'filled');
        scatter(feat2v(1,:),feat2v(2,:),markersize,c,'filled');
%         for i = 1:length(feat1)
%             text(feat1(1,i),feat1(2,i),num2str(i));
%             text(feat2v(1,i),feat2v(2,i),num2str(i));
%         end
    case 3 % lines with true or false
        if exist(sprintf('%s/%s.mat',datapath,view1.filename(1:end-4)),'file')
            load(sprintf('%s/%s.mat',datapath,view1.filename(1:end-4)));
            key1 = pts_coord;
            load(sprintf('%s/%s.mat',datapath,view2.filename(1:end-4)));
            key2 = pts_coord;
        else
            load(sprintf('%s/%s.pts.mat',datapath,view1.filename));
            key1 = d;
            load(sprintf('%s/%s.pts.mat',datapath,view2.filename));
            key2 = d;
        end
        
        F = scatteredInterpolant(key1(1,:)',key1(2,:)',key2(1,:)','natural','nearest');
        feat2p(1,:) = F(feat1(1,:)',feat1(2,:)')';
        F = scatteredInterpolant(key1(1,:)',key1(2,:)',key2(2,:)','natural','nearest');
        feat2p(2,:) = F(feat1(1,:)',feat1(2,:)')';
        dist = sqrt(sum((feat2-feat2p).^2,1));
        idxTrue = dist < th * max(size(view2.img));
        
        for i = find(idxTrue)
            if bg
                plot([ feat1(1,i), feat2v(1,i) ]...
                                ,[ feat1(2,i), feat2v(2,i) ],...
                                '-+','LineWidth',1,'MarkerSize',10,...
                                'color', 'k');
            end
            plot([ feat1(1,i), feat2v(1,i) ]...
                ,[ feat1(2,i), feat2v(2,i) ],...
                '-+','LineWidth',1,'MarkerSize',10,...
                'color', 'b');
        end
        
        for i = find(~idxTrue)
            plot([ feat1(1,i), feat2v(1,i) ]...
                ,[ feat1(2,i), feat2v(2,i) ],...
                '-+','LineWidth',2,'MarkerSize',10,...
                'color', 'r');
        end
end
'''