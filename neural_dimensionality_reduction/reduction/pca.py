from sklearn.decomposition import PCA
    
class PrincipalComponentAnalysis3D():
    def __init__(self):
        self.model = PCA(n_components=3)
    
    def trainAll(self, spikes):
        self.model.fit(spikes)
    
    def transformBatch(self, spikes):
        return self.model.transform(spikes)

