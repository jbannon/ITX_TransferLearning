import numpy as np
import pandas as pd 
import utils
import sys 
import networkx as nx 
import os 
import seaborn as sns
import pickle as pk 

# hallmark_genes = utils.get_hallmark_genes('../data/raw/mdsig_hallmarks.txt')
from typing import List, Tuple, Dict, NamedTuple
import numpy as np 
import pandas as pd
import sys
import argparse
import yaml 

from sklearn.base import BaseEstimator, TransformerMixin
from collections import namedtuple

from utils import ICIDataSet


def process_clinical_data(
	data: pd.DataFrame,
	drop_GBM:bool
	)->pd.DataFrame:
	data['BinarizedResponse'] = data['Response'].apply(lambda x: "R" if x in ['Complete Response','Partial Response'] else "NR")
	data['Response_Class'] = pd.Categorical(data['BinarizedResponse'])
	data['Response_Class'] = data['Response_Class'].cat.codes
	daat = data.dropna(subset=['Response'])
	data = data[data['Sample_Treated'] == False]
	data = data[['Run_ID','TCGA_Tissue','ICI_Rx','Response_Class']]
	data = data[data['TCGA_Tissue']!='GBM']
	return data



def main(config:Dict):

	assert gex_transform in ['tpm','log_tpm','tpm_n','log_tpm_n'], "gex_transform must be one of ['tpm','log_tpm','tpm_n','log_tpm_n']"
	
	drugs = ["Pembro","Atezo","Nivo","Ipi","Ipi + Pembro"]
	
	malignant_exp = pd.read_csv(config['expression_file'],sep = config['expression_sep'])
	measured_genes = list(malignant_expr.columns)[1:]
	

	malignant_phen = pd.read_csv(config['phentoype_file'], sep= config['phenotype_sep'])
	malignant_phen = process_clinical_data(malignant_phen,config['drop_GBM'])
	

	for rx in Rx:
		print("working on treatment {r}".format(r=rx))
		data_dir = config['output_base'] + "{rx}/".format(rx=rx)
		os.makedirs(data_dir,exist_ok = True)






if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-c",help="The config file for these experiments")
	args = parser.parse_args()
	
	with open(args.c) as file:
		config = yaml.safe_load(file)
	

	main(config)
	