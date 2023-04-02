import numpy as np 






class GSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
        GraphShiftOperator:str = None,
        number_of_modes:str = None):
        self.Valid_GSOs=['combinatorial','adjacency','normalized','ranwalk']
        
        assert GraphShiftOperator is not None 
        assert GraphShiftOperator in self.Valid_GSOs


        return self

    def fit(self, adj_matrix:np.array, X=None,y=None):
        self.DegreeMatrix = np.diag(np.sum(adj_matrix),axis=1)
        self.AdjacencyMatrix = adj_matrix
        self.Laplacian = self.DegreeMatrix-self.AdjacencyMatrix
        

        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X


def MakeDiffusionWavelets(
    numScales:int, 
    adjacencyMatrix:np.array
    )->np.array:
    assert len(adjacencyMatrix.shape) == 0
    assert adjacencyMatrix.shape[0] == adjacencyMatrix.shape[0]


class DGScatteringTransform:

    def __init__(self,
        numLayers:int,
        n_scales:int,
        adjaceny_matrix:np.array
        )->None:


        assert numLayers>=0, "Number of layers must be 0 or larger. 0 simply computes a low-pass representation"
        self.numLayers = numLayers

        assert numScales>=0, "Number of scales must be 0 or larger, 0 is just a single step."
        self.numScales = numScales
        if self.numScales==numLayers    