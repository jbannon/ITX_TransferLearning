import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict, Tuple

import numpy as np 
from numpy.linalg import matrix_rank, svd

import pandas as pd 

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.manifold import TSNE
from sklearn.model_selection import ShuffleSplit


from metric_learn import MMC_Supervised, ITML_Supervised, LMNN, NCA 
import pickle as pk 



from utils import ICIDataSet, fetch_STRINGdb,intersect_gene_sets

"""

TODO: create the dataset with the drugs on 

"""

def get_GGI_Network(
	network:str, 
	network_params:Dict
	) -> Tuple[List[str],pd.DataFrame]:
	
	if network.upper()=="STRING":
		genes, GGI = fetch_STRINGdb(**network_params)

	return genes, GGI


def choose_network_params(
	config:Dict
	) -> Tuple[str,Dict[str,Union[str,int]]]:

	network = config['network'].upper()

	if network == "STRING":
		network_params = config['STRING_PARAMS']
	elif network == "PATHWAY COMMONS":
		network_params = config['PWAYCOM_PARAMS']

	return network,network_params


	
def main(
	config:Dict
	):

	drug = config['drug']
	gex_transform = config['gex_transform']
	source_tissue = config['src_tissue']
	target_tissue = config['tgt_tissue']
	network_is_weighted = config['weighted']

	source_data_file = config['input_base']+drug+"/"+source_tissue+"/"+gex_transform+".pickle"
	target_data_file = config['input_base']+drug+"/"+target_tissue+"/"+gex_transform+".pickle"


	print("\nloading source data\n")
	with open(source_data_file, 'rb') as istream:
		sourceDS = pk.load(istream)

	print("loading target data\n")
	with open(target_data_file, 'rb') as istream:
		targetDS = pk.load(istream)
	
	print("loading network info\n")
	network, network_params = choose_network_params(config)

	network_genes, GGI = get_GGI_Network(network, network_params)
	
	common_measured_genes = intersect_gene_sets([targetDS.genes,sourceDS.genes,network_genes])
	print(len(common_measured_genes))
	print(len(network_genes))

	

	
	

	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-c",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.c) as file:
		config = yaml.safe_load(file)
	

	main(config)