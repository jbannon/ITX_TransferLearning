from typing import List,Dict,Union
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from utils import TCGA_NORMAL_MAP
from scipy.stats import variation
import seaborn as sns
import matplotlib.pyplot as plt
import sys


RFC_PARAM_GRID = {'n_estimators':np.arange(10,200,20),
		'criterion':['gini','entropy','log_loss'],
		'max_depth':[2,4,8,16]
		}


LR_PARAM_GRID = {'C':np.logspace(-3,3,100)}
	


	

def fetch_normal_phenotype_info(
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



VALID_MODES = ['tpm','log_tpm','tpm_n','log_tpm_n','uq','deseq']

"""

tpm - just using the transcripts per million from the cancerous samples

log_tpm  - just the tpm for the cancerous samples with log2 transform after pseudocount of 1 added

tpm_n  -  each cancerous sample we subtract off the median tpm from the corresponding tissue

log_tpm_n - each cancerous sample is log2(tpm+1) as are the samples from the corresponding tissues. The median for the normal tissue is calculated and subtracted from the
samples

uq - off-the-shelf upper quartile normalization

deseq - use the median of ratios normalized with a pseudocount


"""

def main(
	mode:str = 'tpm',
	drug:str = 'Atezo'
	):

	mode = mode.lower()
	# normal_tissue_mode = normal_tissue_mode.lower()
	assert drug in ['Atezo','Pembro','Ipi','Nivo','Ipi + Pembro']
	# assert normal_tissue_mode in ['simple','detailed'], "normal_tissue_mode must be one of ['simple','detailed']"
	assert mode in VALID_MODES, "mode must be one of ['tpm','log_tpm','tpm_n','log_tpm_n','uq','deseq']"


	malignant_phenotype_issue = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")

	if mode in ['tpm','log_tpm','tpm_n','log_tpm_n']:

		malignant_expression_measurements = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_tpm.tsv",sep = "\t")
		if mode in ['log_tpm','log_tpm_n']:

			# need to transform normal to log2(tpm+1) for both cases
			
			malignant_expression_measurements.set_index('Run_ID',inplace=True)
			malignant_expression_measurements = np.log2(malignant_expression_measurements + 1)
			malignant_expression_measurements.reset_index(inplace=True,names=['Run_ID'])
			
		
		if mode == 'tpm_n':
		
			normal_expression_measurements = pd.read_csv("../data/raw/GTEx_Medians.gct",sep="\t",skiprows=2)
			normal_tissue_map = None

		elif mode == 'log_tpm_n':
		
			normal_expression_measurements = pd.read_csv("../data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct",sep="\t",skiprows=2)
			normal_tissue_map = None			


		else:
		
			normal_expression_measurements = None
			normal_tissue_map = None
	
	elif mode == 'uq':
	
		malignant_expression_measurements = pd.read_csv("../data/raw/cri/iatlas-ici-genes_norm.tsv",sep = "\t")
		normal_expression_measurements = None
		normal_phenotype_map = None
	
	else: 
	
		# deseq requires raw counts
		malignant_expression_measurements = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_counts.tsv",sep = "\t")
		normal_expression_measurements = None
		normal_phenotype_map = None


	# if mode in ['tpm_n','log_tpm_n']:

	




if __name__ == '__main__':
	main(mode = 'log_tpm')