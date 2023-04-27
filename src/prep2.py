import numpy as np
import pandas as pd 
import utils
import sys 
import os 
import pickle as pk 
import argparse

# hallmark_genes = utils.get_hallmark_genes('../data/raw/mdsig_hallmarks.txt')
from typing import List, Tuple, Dict, NamedTuple
import yaml 

from collections import namedtuple

from utils import ICIDataSet, TCGA_NORMAL_MAP,intersect_gene_sets


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



def write_ICI_DataSet(
	drug:str,
    tissue:str,
    expr_units:str,
    X:np.ndarray,
    y:np.ndarray,
    patient_ids:List[str],
    genes:List[str],
    file_outpath:str
	)->None:
	
	dataset = ICIDataSet(
			drug=drug,
			tissue = tissue,
			expr_units = expr_units,
			X=X,
			y=y,
			patient_ids = patient_ids,
			genes =genes,
			genes_2_idx = {genes[i]:i for i in range(len(genes))}
			)

	with open(file_outpath, 'wb') as ostream:
		pk.dump(dataset, ostream)
	


def main(config:Dict):

	assert config['gex_transform'] in ['tpm','log_tpm','tpm_n','log_tpm_n'], "gex_transform must be one of ['tpm','log_tpm','tpm_n','log_tpm_n']"
	
	drugs = ["Pembro","Atezo","Nivo","Ipi","Ipi + Pembro"]
	
	malignant_exp = pd.read_csv(config['expression_file'],sep = config['expression_sep'])
	measured_genes = list(malignant_exp.columns)[1:]
	if malignant_exp.columns[0] != "Run_ID":
		malignant_exp.columns = ["Run_ID"] + list(malignant_exp.columns[1:])
	

	malignant_phen = pd.read_csv(config['phenotype_file'], sep= config['phenotype_sep'])
	malignant_phen = process_clinical_data(malignant_phen,config['drop_GBM'])
	
	sample_order = malignant_phen['Run_ID'].tolist()

	malignant_phen= malignant_phen.set_index("Run_ID")
	malignant_exp = malignant_exp.set_index("Run_ID")
	
	
	malignant_phen = malignant_phen.loc[sample_order]
	malignant_exp = malignant_exp.loc[sample_order]
	

	if config["gex_transform"] in ["log_tpm","log_tpm_n"]:
		malignant_exp = pd.DataFrame(np.log2(malignant_exp.values+1),index=malignant_exp.index,columns=malignant_exp.columns)
	
	
	malignant_phen.reset_index(inplace=True)
	malignant_phen.rename(columns = {'index':'Run_ID'},inplace=True)

	malignant_exp.reset_index(inplace=True)
	malignant_exp.rename(columns = {'index':'Run_ID'},inplace=True)


	for drug in drugs:
		print("working on treatment {r}".format(r=drug))
		data_dir = config['output_base'] + "{rx}/{t}/".format(rx=drug,t="ALL")
		os.makedirs(data_dir,exist_ok = True)

		treated_samples = malignant_phen[malignant_phen['ICI_Rx']==drug]
		treated_sample_ids = treated_samples['Run_ID'].tolist()
		
		

		X = malignant_exp[malignant_exp['Run_ID'].isin(treated_sample_ids)].drop(columns=['Run_ID']).to_numpy()
		y = treated_samples['Response_Class'].to_numpy()
		
		

		write_ICI_DataSet(
			drug = drug,
			tissue = "ALL",
			expr_units = config['gex_transform'],
			X=X,
			y=y,
			patient_ids = treated_sample_ids,
			genes = list(malignant_exp.columns)[1:],
			file_outpath = data_dir+"{g}.pickle".format(t='ALL',g=config['gex_transform'])
			)
		
	
		
		tissues = list(pd.unique(treated_samples['TCGA_Tissue']))

		for tissue in tissues:
			print("working on tissue: {t}".format(t=tissue))
			if tissue=="GBM":
				print("Skipping GBM")
				continue 
			data_dir = config['output_base'] + "{rx}/{t}/".format(rx=drug,t=tissue)
			os.makedirs(data_dir,exist_ok = True)

			tissue_subset = treated_samples[treated_samples['TCGA_Tissue']==tissue]
			tissue_subset_ids = tissue_subset['Run_ID'].tolist()


			X = malignant_exp[malignant_exp['Run_ID'].isin(tissue_subset_ids)].drop(columns=['Run_ID']).to_numpy()
			y = tissue_subset['Response_Class'].to_numpy()


			write_ICI_DataSet(
				drug = drug,
				tissue = tissue,
				expr_units = config['gex_transform'],
				X=X,
				y=y,
				patient_ids = treated_sample_ids,
				genes = list(malignant_exp.columns)[1:],
				file_outpath = data_dir+"{g}.pickle".format(t=tissue,g=config['gex_transform'])
			)
		

		
			







if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-c",help="The config file for these experiments")
	args = parser.parse_args()
	
	with open(args.c) as file:
		config = yaml.safe_load(file)

	main(config)
	