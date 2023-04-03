import sys
import pandas as pd 
import numpy as np 
from utils import TCGA_NORMAL_MAP

NORMAL_TISSUES = [TCGA_NORMAL_MAP[k][0] for k in TCGA_NORMAL_MAP.keys()]


Normal_Tissue_Medians = pd.read_csv("../data/raw/GTEx_Medians.gct",sep="\t",skiprows=2)
Normal_Tissue_Medians = Normal_Tissue_Medians[["Name","Description"] + NORMAL_TISSUES]
Normal_Tissue_Medians = Normal_Tissue_Medians.drop(columns=['Name'])

Sample_Expression_TPM = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_tpm.tsv",sep="\t")

normal_measured_genes  = list(pd.unique(Normal_Tissue_Medians['Description']))
measured_genes = list(Sample_Expression_TPM.columns)[1:]

gene_intersection = list(set(normal_measured_genes).intersection(set(measured_genes)))
gene_intersection.sort()

Normal_Tissue_Medians = Normal_Tissue_Medians[Normal_Tissue_Medians['Description'].isin(gene_intersection)]
Normal_Tissue_Medians = Normal_Tissue_Medians.groupby(['Description']).median().reset_index()


Sample_Clinical = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")
Sample_Clinical = Sample_Clinical[['Run_ID','TCGA_Tissue']]


normalized_df = pd.DataFrame()

for tissue in list(pd.unique(Sample_Clinical['TCGA_Tissue'])):
	if tissue == 'GBM':
		continue
	print(tissue)
	Tissue_Sample_IDs = list(Sample_Clinical[Sample_Clinical['TCGA_Tissue']==tissue].to_numpy()[:,0])

	cancer_samples = Sample_Expression_TPM[Sample_Expression_TPM['Run_ID'].isin(Tissue_Sample_IDs)]

	normal_expression = Normal_Tissue_Medians[TCGA_NORMAL_MAP[tissue][0]].to_numpy()
	

	cancer_samples = cancer_samples[gene_intersection].to_numpy()
	cancer_samples_norm = cancer_samples - normal_expression

	temp_df = pd.DataFrame(cancer_samples_norm,columns = gene_intersection)
	temp_df['Run_ID'] = Tissue_Sample_IDs
	normalized_df = pd.concat([normalized_df,temp_df],axis=0)


normalized_df = normalized_df[['Run_ID']+gene_intersection]
print(normalized_df.head())
print(normalized_df.shape)
normalized_df.to_csv("../data/preprocessed/NMS_Scaled_TPM_Expression.csv",index=False)
	

	
