
import sys
import pandas as pd 
import numpy as np 
from utils import TCGA_NORMAL_MAP

NORMAL_TISSUES = [TCGA_NORMAL_MAP[k][0] for k in TCGA_NORMAL_MAP.keys()]


normal_sample_info = pd.read_csv("../data/raw/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",sep="\t")
normal_sample_info = normal_sample_info[['SAMPID','SMTS','SMTSD']]
normal_sample_info.columns= ['SAMPID','TISSUE','TISSUE_DETAILED']
normal_sample_info = normal_sample_info[normal_sample_info['TISSUE_DETAILED'].isin(NORMAL_TISSUES)]

normal_expression_measurements = pd.read_csv("../data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct",sep="\t",skiprows=2)





columns = {}

for tissue in NORMAL_TISSUES:
	tissue_samples = list(pd.unique(normal_sample_info[normal_sample_info['TISSUE_DETAILED']==tissue]['SAMPID']))
	common_samples = list( set(normal_expression_measurements.columns).intersection(set(tissue_samples)))

	
	transformed_tissue_measurements = np.log2(normal_expression_measurements[common_samples].values+1)
	median_expression = np.median(transformed_tissue_measurements,axis=1)
	columns[tissue] = median_expression

columns['Description'] = normal_expression_measurements['Description']

median_df = pd.DataFrame(columns)
median_df.to_csv("../data/preprocessed/log2po_normal_medians.csv")