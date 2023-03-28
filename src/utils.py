from typing import List, Tuple, Dict
import numpy as np 
import pandas as pd
import sys
from sklearn.base import BaseEstimator, TransformerMixin

def get_hallmark_genes(
    file_path:str
    )->List[str]:
    
    gene_set = []
    with open(file_path,"r") as f:
        lines = f.readlines()

    for line in lines:
        temp = line.split("\t")
        temp = [x.rstrip() for x in temp ]
        temp = temp[2:]
        gene_set = gene_set + temp

    gene_set = list(set(gene_set))
    return gene_set



def fetch_pathway_commons_network(
    file_path:str,
    interaction_type = "interacts-with",
    filter_gene_lists:List[List[str]]=None,
    filter_genes:bool = False,
    symmetrize_interaction:bool = True,
    ) -> Tuple[List[str],Dict[str,int],Dict[int,str],np.array]:

    interaction = pd.read_csv(file_path,sep="\t",header=None,names = ['A','Interaction-Type','B'])
    interaction = interaction[interaction['Interaction-Type']==interaction_type]

    node_genes = list(set(interaction['A']).union(set(interaction['B'])))

    if filter_genes and filter_gene_lists is not None:
        node_genes = intersect_gene_sets([node_genes]+filter_gene_lists)

    
    interaction =interaction[ (interaction['A'].isin(node_genes)) & (interaction['B'].isin(node_genes))]
    
    n_genes = len(node_genes)
    
    gene_2_idx = {}
    idx_2_gene = {}

    for i in range(len(node_genes)):
        gene_name  = node_genes[i]
        gene_2_idx[gene_name]=i 
        idx_2_gene[i] = gene_name

    adjacency_matrix = np.zeros((n_genes,n_genes))
    
    for idx,row in interaction.iterrows():

        adjacency_matrix[gene_2_idx[row['A']],gene_2_idx[row['B']]]=1
        if symmetrize_interaction:
            adjacency_matrix[gene_2_idx[row['B']],gene_2_idx[row['A']]]=1

    





    return node_genes,gene_2_idx,idx_2_gene, adjacency_matrix





def intersect_gene_sets(
    gene_sets:List[List[str]]
    ) -> List[str]:
    
    result = set(gene_sets[0])
    for i in range(1,len(gene_sets)):
        result = result.intersection(set(gene_sets[i]))

    return list(result)

    


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

#     