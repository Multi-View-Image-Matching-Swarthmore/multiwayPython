class pairwiseMatches:
    def __init__(self, matchInfo, nFeature, Xraw, X):
        self.matchInfo = matchInfo
        self.nFeature = nFeature
        self.Xraw = Xraw
        self.X = X
        self.filename = None

class jointMatchInfo:
    def __init__(self, eigV, nFeature, filename, time, Z):
        self.eigV = eigV
        self.nFeature = nFeature
        self.filename = filename
        self.time = time
        self.Z = Z
