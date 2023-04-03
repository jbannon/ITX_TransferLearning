import numpy as np
import pandas as pd 
import utils
from utils import NORMAL_TISSUES
import sys 
import networkx as nx 
import os 


# hallmark_genes = utils.get_hallmark_genes('../data/raw/mdsig_hallmarks.txt')
Rx = ["Pembro","Atezo","Nivo","Ipi","Ipi + Pembro","Ipi + Nivo"]



expr_prenormalized = pd.read_csv('../data/raw/cri/iatlas-ici-genes_norm.tsv',sep="\t")
expr_NMS = pd.read_csv("../data/preprocessed/NMS_Scaled_TPM_Expression.csv",sep="\t")


measured_genes = list(GEX_Features.columns)

network_genes, gene_2_idx, idx_2_gene, adjacency_matrix = utils.fetch_pathway_commons_network(
	file_path = "./data/raw/PathwayCommons12.All.hgnc.sif", 
	filter_gene_lists=[hallmark_genes,measured_genes],
	filter_genes=True)



G = nx.from_numpy_array(adjacency_matrix)

ccs = nx.connected_components(G)
isolated_genes = []

for cc in ccs: 
	if len(cc)==1:
		isolated_genes.append(idx_2_gene[list(cc)[0]])


GEX_Features = GEX_Features[['sample']+network_genes]

sorted_indices = sorted(list([x for x in idx_2_gene.keys()]))
genes_in_order = [idx_2_gene[j] for j in sorted_indices]



for rx in Rx: 
	os.makedirs("./data/preprocessed/{rx}".format(rx=rx),exist_ok=True)
	response_data = pd.read_csv("./data/summaries/{r}.csv".format(r=rx),index_col=0)
	response_data = response_data.dropna(subset=['PFS_e','PFS_d'])
	response_data['event'] = response_data['PFS_e'].apply(lambda x: int(x))
	Rx_GEX = GEX_Features[GEX_Features['sample'].isin(list(pd.unique(response_data['Run_ID'])))]
	tissues = list(pd.unique(response_data['TCGA_Tissue']))

	for tissue in tissues:
		tissue_df = response_data[response_data['TCGA_Tissue']==tissue]
		tissue_df = tissue_df[['Run_ID','Response','event','PFS_d']]

		fname = "./data/preprocessed/{rx}/{rx}_{tissue}_{src}{dst}.npz".format(rx=rx,tissue=tissue,src='C',dst='S')

		full_dataset = Rx_GEX.merge(tissue_df,left_on='sample',right_on='Run_ID')
		features  = full_dataset.drop(columns=['sample','Run_ID','Response','event','PFS_d'])
		
		features = features[genes_in_order]
		X = features.values
		Y_surv = full_dataset[['PFS_d','event']].values
		full_dataset['Response_Class'] = pd.Categorical(full_dataset['Response'])
		full_dataset['Response_Class'] = full_dataset['Response_Class'].cat.codes

		class_code_map = full_dataset[['Response','Response_Class']].values
		Y_class = class_code_map[:,1]
		Y_class = Y_class.reshape((-1,)).astype('int64')
		print(Y_class)
		np.savez(fname, X=X, Y_s =Y_surv, Y_c = Y_class,adj_mat=adjacency_matrix)







