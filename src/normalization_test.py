import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import sys
from utils import TCGA_NORMAL_MAP
from scipy.stats import variation
import seaborn as sns
import matplotlib.pyplot as plt

def fetch_normal_sample_tissues(
	file_path:str = "../data/raw/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",sep="\t"
	) -> pd.DataFrame:
	

	df = pd.read_csv(file_path,sep="\t")
	df = df[['SAMPID','SMTS','SMTSD']]
	df.columns = ['SAMPID','TISSUE','TISSUE_DETAILED']


	return df

def median_of_ratios_normalize(
	X:np.array,
	sample_axis:int = 1,
	pseudo_count:int = 1
	)->np.array:
	

	""" Implements the DESEQ2 Normalization Procedure
	
			NB: This procedure expects Raw Counts
	""" 

	assert sample_axis in [0,1], "sample axis must be an integer and must be either zero or 1"
	assert len(X.shape)==2, "X must be a 2d numpy array"

	reference_sample = np.prod(X+pseudo_count,axis=sample_axis)**1.0/(X.shape[sample_axis])
	print(reference_sample)
	size_factors = np.divide(X,reference_sample)
	print(size_factors)




def fetch_phenotype_info(
	file_path:str = "../data/raw/cri/iatlas-ici-sample_info.tsv",
	keep_cols:List[str] = ['Run_ID','TCGA_Tissue']
	)->pd.DataFrame:

	df = pd.read_csv(file_path,sep="\t")
	df = df[df['Sample_Treated']==False]

	return df[keep_cols]


def fetch_genes_by_annotation(
	df:pd.DataFrame,
	ann_col:str,
	gene_col:str,
	anno:str
	)->List[str]:
	
	subset = df[df[ann_col]==anno]
	genes = pd.unique(subset[gene_col]).tolist()

	return genes


def make_geneset_dictionary(
	file_path:str = "../data/raw/GO_terms.csv"
	) ->Dict[str,List[str]]:
	
	GO_df = pd.read_csv(file_path)

	genesets = {
			'CC_Adh':fetch_genes_by_annotation(df=GO_df, gene_col='Symbol',ann_col='Annotated Term',anno='cell-cell adhesion'),
			'C_Adh':fetch_genes_by_annotation(df=GO_df, gene_col='Symbol',ann_col='Annotated Term',anno='cell adhesion'),
			# 'rRNA':['5S_rRNA','5_8S_rRNA'],
			'HKG': ['C1orf43','CHMP2A', 'EMC7', 'GPI','PSMB2', 'PSMB4', 'RAB7A', 'REEP5', 'SNRPD3', 'VCP', 'VPS29'] #from genemunge; https://www.tau.ac.il/~elieis/HKG/
				}

	genesets['CC_Adh'] = [x.upper() for x in genesets['CC_Adh']]
	genesets['C_Adh'] = [x.upper() for x in genesets['C_Adh']]
	all_adhesion_genes = list(set(genesets['CC_Adh']).union(set(genesets['C_Adh'])))
	genesets['Adh'] = all_adhesion_genes


	return genesets



def main():
	ICI_root = "../data/raw/cri/"
	NORM_root = "../data/preprocessed/"
	genesets = make_geneset_dictionary()
	phenotype_df = fetch_phenotype_info()
	kept_ICI_ids = pd.unique(phenotype_df['Run_ID']).tolist()
	NORM_TISSUE_MAP = fetch_normal_sample_tissues()
	
	

	# ICI_TPM = pd.read_csv(ICI_root + "iatlas-ici-hgnc_tpm.tsv",sep="\t")
	# ICI_TPM = ICI_TPM[ICI_TPM['Run_ID'].isin(kept_ICI_ids)]
	# ICI_TPM = ICI_TPM.set_index('Run_ID')
	
	print("fetching ICI Count Data")
	ICI_COUNTS = pd.read_csv(ICI_root + "iatlas-ici-hgnc_counts.tsv",sep="\t")
	print(ICI_COUNTS.columns)
	
	print("fetching normal count data")

	NORM_COUNTS = pd.read_csv(NORM_root+"normal_counts.csv")
	
	normal_samp_counts = {}
	rows = {'variation':[],'geneset':[],'tissue':[],'sample_type':[],'normalization':[]}

	for tissue in list(pd.unique(phenotype_df['TCGA_Tissue'])):

		if tissue=="GBM":
			continue


		tissue_ids = list(pd.unique(phenotype_df[phenotype_df['TCGA_Tissue']==tissue]['Run_ID']))
		cancer_counts_df = ICI_COUNTS[ICI_COUNTS['Run_ID'].isin(tissue_ids)]

		nt = TCGA_NORMAL_MAP[tissue][0]
		normal_samps = NORM_TISSUE_MAP[NORM_TISSUE_MAP['TISSUE_DETAILED']==TCGA_NORMAL_MAP[tissue][0]]
		normal_samp_ids = list(pd.unique(normal_samps['SAMPID']))
		normal_cols = list(set(['Description']+normal_samp_ids).intersection(set(NORM_COUNTS.columns)))
	
		temp_norm = NORM_COUNTS[normal_cols]

		normal_samp_counts[tissue]=temp_norm.shape[1]
		
		for gs in genesets.keys():
			genes = genesets[gs]
			cancer_measured_genes = set(genes).intersection(set(cancer_counts_df.columns))
			norm_measured_genes = temp_norm[temp_norm['Description'].isin(genes)]
			norm_measured_genes = set(pd.unique(norm_measured_genes['Description']))
			common_measured_genes = list(cancer_measured_genes.intersection(norm_measured_genes))
			
			geneset_norm = temp_norm[temp_norm['Description'].isin(common_measured_genes)].drop(columns=['Description'])
			# normal has measured genes as rows, samples as columns so we need to transpose
			print(geneset_norm)
			normal_counts = geneset_norm.values.T

			cancer_counts = cancer_counts_df[common_measured_genes].values
			
			print(tissue)
			print(gs)
			assert cancer_counts.shape[1]==normal_counts.shape[1]
			#both should be nsamples x ngenes
			
			cancer_var = variation(cancer_counts,axis=0)
			normal_var = variation(normal_counts,axis=0)
			rows['variation'].extend(list(cancer_var))
			rows['sample_type'].extend(['cancer']*cancer_var.shape[0])
			rows['variation'].extend(list(normal_var))
			rows['sample_type'].extend(['normal']*cancer_var.shape[0])
			rows['tissue'].extend([tissue]*(2*cancer_var.shape[0]))
			rows['geneset'].extend([gs]*(2*cancer_var.shape[0]))
			rows['normalization'].extend(['raw_count']*(2*cancer_var.shape[0]))
			
			

			normal_mean = np.mean(normal_counts,axis=0)
			
			# normal_median = np.median(normal_counts,axis=0)
			# normal_std = np.std(normal_counts,axis=0)
			# cancer_mean_adj= cancer_counts - normal_mean
			# cancer_median_adj = cancer_counts - normal_median 
			# cancer_mean_z = cancer_mean_adj/normal_std
			# cancer_median_z = cancer_median_adj/normal_std

	results = pd.DataFrame(rows)
	print(results)
	results.to_csv("../data/results/count_variation.csv")
	
	

	
	
	
	
	
if __name__ == '__main__':
	main()