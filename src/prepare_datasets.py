import numpy as np
import pandas as pd 
import utils
import sys 
import networkx as nx 
import os 
import seaborn as sns
import pickle as pk 

# hallmark_genes = utils.get_hallmark_genes('../data/raw/mdsig_hallmarks.txt')

from utils import ICIDataset


Rx = ["Pembro","Atezo","Nivo","Ipi","Ipi + Pembro","Ipi + Nivo"]

NormalizationMethods = ['TPM','LogTPM','LogTPM-Median','LogTPM-Centered','UQ','LogTPM-NormalMedian', 'LogTPM-NormalMedianFull']



expr_UQ = pd.read_csv('../data/raw/cri/iatlas-ici-genes_norm.tsv',sep="\t")
expr_TPM = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_tpm.tsv",sep = "\t")







clinical_data = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv", sep= "\t")
clinical_data['BinarizedResponse'] = clinical_data['Response'].apply(lambda x: "R" if x in ['Complete Response','Partial Response'] else "NR")
clinical_data['Response_Class'] = pd.Categorical(clinical_data['BinarizedResponse'])
clinical_data['Response_Class'] = clinical_data['Response_Class'].cat.codes
clinical_data = clinical_data.dropna(subset=['Response'])
clinical_data = clinical_data[clinical_data['Sample_Treated'] == False]
clinical_data = clinical_data[['Run_ID','TCGA_Tissue','ICI_Rx','Response_Class']]
clinical_data = clinical_data[clinical_data['TCGA_Tissue']!='GBM']
sample_order = clinical_data['Run_ID'].tolist()


clinical_data = clinical_data.set_index("Run_ID")
expr_NMA = expr_NMA.set_index("Run_ID")
expr_prenormalized = expr_prenormalized.set_index("sample")

print("reordering samples")
# enforce order of samples so that features and response are properly aligned

clinical_data = clinical_data.loc[sample_order]
expr_NMA = expr_NMA.loc[sample_order]
expr_prenormalized = expr_prenormalized.loc[sample_order]

clinical_data.reset_index(inplace=True)
clinical_data.rename(columns = {'index':'Run_ID'},inplace=True)

expr_NMA.reset_index(inplace=True)
expr_NMA.rename(columns = {'index':'Run_ID'},inplace=True)

expr_prenormalized.reset_index(inplace=True)
expr_prenormalized.rename(columns = {'sample':'Run_ID'},inplace=True)

expr_NMA_genes = list(expr_NMA.columns)[1:]
NMA_idx_2_gene = {k:expr_NMA_genes[k] for k in range(len(expr_NMA_genes))}

expr_prenormalized_genes = list(expr_prenormalized.columns)[1:]
prenorm_idx_2_gene = {k:expr_prenormalized_genes[k] for k in range(len(expr_prenormalized_genes))}

with open ("../data/preprocessed/nma_idx_2_gene.pk",'wb') as ostream:
	pk.dump(NMA_idx_2_gene,ostream)


with open ("../data/preprocessed/prenorm_idx_2_gene.pk",'wb') as ostream:
	pk.dump(prenorm_idx_2_gene,ostream)

for rx in Rx: 

	print("working on treatment {r}".format(r=rx))

	data_dir = "../data/preprocessed/{rx}/".format(rx=rx)
	fig_dir = "../figs/{rx}/exploratory/manifold_embeddings/".format(rx=rx)
	os.makedirs(data_dir,exist_ok = True)
	os.makedirs(fig_dir, exist_ok = True)

	
	rx_data = clinical_data[clinical_data['ICI_Rx']==rx]
	
	rx_ids = rx_data['Run_ID'].tolist()
	
	response = rx_data['Response_Class'].to_numpy()
	
	
	rx_NMA_data = expr_NMA[expr_NMA['Run_ID'].isin(rx_ids)].drop(columns=['Run_ID']).to_numpy()

	rx_norm_data = expr_prenormalized[expr_prenormalized['Run_ID'].isin(rx_ids)].drop(columns=['Run_ID']).to_numpy()
	
	np.savez(data_dir+"ALL_NMA.npz",X=rx_NMA_data,y=response)
	np.savez(data_dir+"ALL_prenorm.npz",X=rx_norm_data,y=response)

	

	tissues = list(pd.unique(rx_data['TCGA_Tissue']))
	for tissue in tissues:
		print("working on tissue: {t}".format(t=tissue))
		if tissue=="GBM":
			continue 

		tissue_subset = rx_data[rx_data['TCGA_Tissue']==tissue]
		tissue_ids = tissue_subset['Run_ID'].tolist()

		tissue_response = tissue_subset['Response_Class'].to_numpy()
		tissue_NMA_data = expr_NMA[expr_NMA['Run_ID'].isin(tissue_ids)].drop(columns=['Run_ID']).to_numpy()
		tissue_norm_data = expr_prenormalized[expr_prenormalized['Run_ID'].isin(tissue_ids)].drop(columns=['Run_ID']).to_numpy()
		np.savez(data_dir+"{t}_NMA.npz".format(t=tissue.upper()),X=tissue_NMA_data,y=tissue_response)
		np.savez(data_dir+"{t}_prenorm.npz".format(t=tissue.upper()),X=tissue_norm_data,y=tissue_response)




		
	

if __name__ == '__main__':
	main()




