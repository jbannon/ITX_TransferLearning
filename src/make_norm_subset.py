"""
Subset normal genes for future operation speedup

"""

from utils import TCGA_NORMAL_MAP,TCGA_NORMAL_MAP_Plus
import pandas as pd
import numpy as np 
import sys
from 

def fetch_genes_by_annotation(
	df:pd.DataFrame,
	ann_col:str,
	gene_col:str,
	anno:str
	)->List[str]:
	
	subset = df[df[ann_col]==anno]
	genes = pd.unique(subset[gene_col]).tolist()

	return genes




def fetch_normal_sample_tissues(
	file_path:str = "../data/raw/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",sep="\t"
	) -> pd.DataFrame:
	

	df = pd.read_csv(file_path,sep="\t")
	df = df[['SAMPID','SMTS','SMTSD']]
	df.columns = ['SAMPID','TISSUE','TISSUE_DETAILED']


	return df

def transpose_df(
	df:pd.DataFrame,
	index_col:str = 'Description',
	rows_name:str = 'SAMPID')->pd.DataFrame:
	new_col_labels = list(df[index_col])
	df = df.set_index("Description")
	print("in fn")
	print(df.shape)
	new_row_labels = list(df.columns)
	new_df = pd.DataFrame(df.values.T,columns = new_col_labels)
	new_df[rows_name] = new_row_labels
	new_df = new_df[[rows_name]+new_col_labels]
	return new_df	



	
NORM_MAP = fetch_normal_sample_tissues()
TISSUES = []
for v in TCGA_NORMAL_MAP.values():
	for j in v:
		TISSUES.append(j)
print(TISSUES)

TISSUES_DETAILED = []
for v in TCGA_NORMAL_MAP_Plus.values():
	for j in v:
		TISSUES_DETAILED.append(j)
print(TISSUES_DETAILED)

nt = NORM_MAP[NORM_MAP['TISSUE_DETAILED'].isin(TISSUES)]
nt_detailed = NORM_MAP[NORM_MAP['TISSUE_DETAILED'].isin(TISSUES_DETAILED)]
nt_ids = list(pd.unique(nt['SAMPID']))
ntd_ids = list(pd.unique(nt_detailed['SAMPID']))

NORM_COUNTS = pd.read_csv("../data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct",sep="\t",skiprows=2)

nt_ids = list(set(nt_ids).intersection(set(NORM_COUNTS.columns)))
ntd_ids = list(set(ntd_ids).intersection(set(NORM_COUNTS.columns)))

nt_counts = NORM_COUNTS[['Description']+nt_ids]
# nt_counts = transpose_df(nt_counts)
# genesets = make_geneset_dictionary()


nt_counts.to_csv("../data/preprocessed/normal_counts.csv",index=False)


