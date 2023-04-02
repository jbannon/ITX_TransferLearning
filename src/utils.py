from typing import List, Tuple, Dict
import numpy as np 
import pandas as pd
import sys
from sklearn.base import BaseEstimator, TransformerMixin


TCGA_NORMAL_MAP = {
    'SKCM':['Skin - Sun Exposed (Lower leg)'], 
    'STAD':['Stomach'],
    'KIRC':['Kidney - Cortex'],
    'BLCA':['Bladder']
    }



TCGA_NORMAL_MAP_Plus = {
    'SKCM':['Skin - Not Sun Exposed (Suprapubic)', 'Skin - Sun Exposed (Lower leg]'], 
    'STAD':['Stomach'],
    'KIRC':['Kidney - Cortex','Kidney - Medulla'],
    'BLCA':['Bladder']
    }

NANOSTRING_STUDIES = ['Chen 2016','Melero 2019','Prat 2017']


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



def fetch_normal_expression_medians(
    file_path:str,
    tissues_to_keep:List[str] = ['Skin - Not Sun Exposed (Suprapubic)', 
    'Skin - Sun Exposed (Lower leg)','Stomach','Kidney - Cortex','Kidney - Medulla','Bladder']
    )-> pd.DataFrame:
    

    median_expression = pd.read_csv(file_path,sep="\t",skiprows=2)
    if tissues_to_keep is not None:
        median_expression = median_expression[["Name","Description"] + tissues_to_keep]
    
    return median_expression


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

    


def collect_GeneMANIA_genes(
    file_path:str = "../data/raw/networks/GeneMANIA/identifier_mappings.txt"
    ) -> List[str]:
    gene_gene_interactions = pd.read_csv(file_path, sep="\t")
    gene_names = list(pd.unique(gene_gene_interactions['Name']))
    return gene_names
    


def collect_PathwayCommons_genes(
    file_path:str = "../data/raw/networks/PathwayCommons/PathwayCommons12.All.hgnc.sif",
    interaction_type:str = 'interacts-with'
    ) -> List[str]:
    gene_gene_interactions = pd.read_csv(file_path,sep="\t",header=None,names = ['A','Interaction-Type','B'])
    gene_gene_interactions = gene_gene_interactions[gene_gene_interactions['Interaction-Type']==interaction_type]
    src_genes = set(pd.unique(gene_gene_interactions['A']))
    dst_genes = set(pd.unique(gene_gene_interactions['B']))
    gene_names = list(src_genes.union(dst_genes))
    return gene_names

def collect_STRINGdb_genes(
    interaction_file_path:str = "../data/raw/networks/STRINGdb/9606.protein.links.v11.5.txt",
    annotation_file_path:str = "../data/raw/networks/STRINGdb/9606.protein.info.v11.5.txt",
    score_threshold:int = 700
    ) ->List[str]:
    gene_gene_interactions = pd.read_csv(interaction_file_path,sep=" ")
    annotations = pd.read_csv(annotation_file_path,sep="\t")
    gene_gene_interactions = gene_gene_interactions[gene_gene_interactions['combined_score']>=score_threshold]
    protein_ids = list(set(pd.unique(gene_gene_interactions['protein1'])).union(set(pd.unique(gene_gene_interactions['protein2']))))
    annotations = annotations[annotations['#string_protein_id'].isin(protein_ids)]
    gene_names = list(pd.unique(annotations['preferred_name']))

    return gene_names
    

