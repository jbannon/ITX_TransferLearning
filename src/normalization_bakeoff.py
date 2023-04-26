from typing import List,Dict,Union
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, KFold,LeaveOneOut, StratifiedKFold,cross_val_score
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler,MaxAbsScaler, MinMaxScaler
from utils import TCGA_NORMAL_MAP
from sklearn.base import RegressorMixin
import sys
import os 


RFC_PARAM_GRID = {
	'clf__n_estimators':np.arange(10,100,50),
	'clf__criterion':['gini','entropy','log_loss'],
	'clf__max_depth':[2,4,8,None]
	}


LR_PARAM_GRID = {
	'clf__C':np.logspace(-3,3,1000)
	# 'clf__max_iter':[1000,5000,10000]
	}


SVC_PARAM_GRID = {
	'clf__C':np.logspace(-3,3,1000),
	'clf__kernel':['linear','rbf','poly'],
	'clf__degree':[1,2,3]
	}


ESTIMATOR_2_GRID = {'RFC':RFC_PARAM_GRID,'LR':LR_PARAM_GRID,'SVC':SVC_PARAM_GRID}


VALID_MODES = ['tpm','log_tpm','tpm_n','log_tpm_n','uq','deseq']
VALID_DRUGS = ['Atezo','Pembro','Ipi','Nivo','Ipi + Pembro']
VALID_ESTIMATORS = ['RFC','SVC','LR']


MODEL_TUPLES = {
	'RFC':('clf',RandomForestClassifier()),
	'LR':('clf',LogisticRegression(max_iter=5000)),
	'SVC':('clf',SVC())
	}


PIPELINE_DICT = {'None':[], 
		'Mean':[('scaler',StandardScaler(with_std=False))],
		'Standardized':[('scaler',StandardScaler())],
		'Median':[('scaler',RobustScaler(with_scaling=False))],
		'RobustScaled':[('scaler',RobustScaler())]
		# 'MaxAbs':[('scaler',MaxAbsScaler())],
		# 'MinMax':[('scaler',MinMaxScaler())]
		}



def subtract_normal_median_from_samples(
	malignant_expr:pd.DataFrame,
	malignant_phen:pd.DataFrame,
	normal_expr:pd.DataFrame,
	tissue_map:Dict[str,List[str]]
	)->pd.DataFrame:

	print("subtracting off the median normal expression")	

	normal_measured_genes  = list(pd.unique(normal_expr['Description']))
	measured_genes = list(malignant_expr.columns)[1:]

	gene_intersection = list(set(normal_measured_genes).intersection(set(measured_genes)))
	gene_intersection.sort()


	normal_expr = normal_expr[normal_expr['Description'].isin(gene_intersection)]
	normal_expr = normal_expr.groupby(['Description']).median().reset_index()
	normal_expr.set_index('Description',inplace=True)
	normal_expr = normal_expr.loc[gene_intersection]
	normal_expr = normal_expr.reset_index()

	malignant_expr = malignant_expr[['Run_ID']+gene_intersection]

	l1 = list(normal_expr['Description'])
	l2 = list(malignant_expr.columns[1:])
	for i in range(len(l1)):
		if l1[i] != l2[i]:
			print("error at {f}".format(f=l1[i]))
	print('cleared order check')


	normalized_df = pd.DataFrame()

	for tissue in list(pd.unique(malignant_phen['TCGA_Tissue'])):
		if tissue == 'GBM':
			continue

		malignant_sample_ids = list(malignant_phen[malignant_phen['TCGA_Tissue']==tissue].to_numpy()[:,0])
		
		malignant_samples = malignant_expr[malignant_expr['Run_ID'].isin(malignant_sample_ids)]

		normal_expression = normal_expr[tissue_map[tissue][0]].to_numpy()
		# print(malignant_expression)

		malignant_samples = malignant_samples[gene_intersection].to_numpy()
		malignant_samples = malignant_samples - normal_expression

		temp_df = pd.DataFrame(malignant_samples,columns = gene_intersection)
		temp_df['Run_ID'] = malignant_sample_ids
		normalized_df = pd.concat([normalized_df,temp_df],axis=0)

	normalized_df = normalized_df[['Run_ID']+gene_intersection]
	
	return normalized_df


	

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
	estimator:str = 'RFC',
	drug:str = 'Pembro',
	k_outer:int = 10,
	k_inner:int = 4,
	):
	
	write_res_path = "../results/{rx}/{norm}/".format(rx=drug,norm=mode)
	os.makedirs(write_res_path,exist_ok=True)
	
	mode = mode.lower()
	assert estimator in VALID_ESTIMATORS, "estimator must be one of ['RFC','SVC','LR'] "
	# normal_tissue_mode = normal_tissue_mode.lower()
	assert drug in VALID_DRUGS, "drug must be one of ['Atezo','Pembro','Ipi','Nivo','Ipi + Pembro']"
	# assert normal_tissue_mode in ['simple','detailed'], "normal_tissue_mode must be one of ['simple','detailed']"
	assert mode in VALID_MODES, "mode must be one of ['tpm','log_tpm','tpm_n','log_tpm_n','uq','deseq']"

	NORMAL_TISSUES = [TCGA_NORMAL_MAP[k][0] for k in TCGA_NORMAL_MAP.keys()]

	malignant_phenotype = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")
	normal_expression = None

	if mode in ['tpm','tpm_n','log_tpm','log_tpm_n']:

		print('reading in malignant TPM data')
		malignant_expression = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_tpm.tsv",sep = "\t")

		if mode in ['log_tpm','log_tpm_n']:

			# need to transform normal to log2(tpm+1) for both cases
			print('log2(TPM+1) transforming malignant expression')
			malignant_expression.set_index('Run_ID',inplace=True)
			malignant_expression = np.log2(malignant_expression + 1)
			malignant_expression.reset_index(inplace=True,names=['Run_ID'])
			
		if mode == 'tpm_n':
			print('reading normal median TPM')
			normal_expression = pd.read_csv("../data/raw/GTEx_Medians.gct",sep="\t",skiprows=2)
			normal_expression = normal_expression[["Description"] + NORMAL_TISSUES]


		elif mode == 'log_tpm_n':
			print('reading normal medain log2(TPM+1)')
			normal_expression = normal_expression = pd.read_csv("../data/preprocessed/log2po_normal_medians.csv",index_col=0)
			
		
		

	elif mode == 'uq':
		print('reading upper quartile normed counts')
		malignant_expression = pd.read_csv("../data/raw/cri/iatlas-ici-genes_norm.tsv",sep = "\t")
		new_cols = list(malignant_expression.columns)
		new_cols[0]="Run_ID"
		malignant_expression.columns = new_cols

		

	else: 

		# deseq requires raw counts
		print("reading raw counts")
		malignant_expression = pd.read_csv("../data/raw/cri/iatlas-ici-hgnc_counts.tsv",sep = "\t")
		
	if normal_expression is not None:
		malignant_expression = subtract_normal_median_from_samples(malignant_expression,malignant_phenotype,normal_expression,TCGA_NORMAL_MAP)

	
	malignant_phenotype['BinarizedResponse'] = malignant_phenotype['Response'].apply(lambda x: "R" if x in ['Complete Response','Partial Response'] else "NR")
	malignant_phenotype['Response_numeric'] = pd.Categorical(malignant_phenotype['BinarizedResponse'])
	malignant_phenotype['Response_numeric'] = malignant_phenotype['Response_numeric'].cat.codes

	malignant_phenotype = malignant_phenotype[ (malignant_phenotype['ICI_Rx']==drug) & (malignant_phenotype['Sample_Treated']==False)]

	tissues = list(pd.unique(malignant_phenotype['TCGA_Tissue']))
	tissues.append('ALL')

	
	
	for k in PIPELINE_DICT.keys():
		PIPELINE_DICT[k].append(MODEL_TUPLES[estimator])
		PIPELINE_DICT[k]= Pipeline(PIPELINE_DICT[k])
	
	

	GEX_units = []
	trial = []
	model_type = []
	scores = []
	preproc = []
	tissue_type = []

	
	for tissue in tissues:
		print("Starting Tisssue {t}".format(t=tissue))
		if tissue!='ALL':
			tissue_subset = malignant_phenotype[malignant_phenotype['TCGA_Tissue']==tissue]
		else:
			tissue_subset = malignant_phenotype

		sample_ids = list(tissue_subset['Run_ID'])
		
		malignant_expression.set_index('Run_ID',inplace=True)
		malignant_subset = malignant_expression.loc[sample_ids]
		malignant_expression.reset_index(inplace=True)

		X = malignant_subset.values
		y = tissue_subset['Response_numeric'].values

		
		
		for pipe_name in PIPELINE_DICT.keys():
			
			pipe_estimator = PIPELINE_DICT[pipe_name]
			print("\n Evaluating Preprocessing Step: {p} with tissue {t}\n".format(p=pipe_name,t=tissue))
			
			
		

			outer_cv = StratifiedKFold(n_splits=k_outer,shuffle=True,random_state=1234)
			inner_cv = StratifiedKFold(n_splits=k_inner,shuffle=True,random_state=1234)
			
			classifier = GridSearchCV(estimator=pipe_estimator, scoring='balanced_accuracy',param_grid = RFC_PARAM_GRID, cv=inner_cv)
			nested_scores = cross_val_score(classifier, scoring='balanced_accuracy',X=X, y=y, cv=outer_cv,verbose=4)
			
	
			
			
			GEX_units.extend([mode]*k_outer)
			trial.extend([i for i in range(k_outer)])
			model_type.extend([estimator]*k_outer)
			preproc.extend([pipe_name]*k_outer)
			tissue_type.extend([tissue]*k_outer)
			scores.extend(nested_scores)

	results = pd.DataFrame({
		'GEX_units':GEX_units,
		'Trial':trial,
		'Estimator':model_type,
		'Scores':scores,
		'Preprocessing':preproc,
		'Tissue':tissue_type
		})

	
	results.to_csv(write_res_path+"{m}.csv".format(m=estimator))

	





		
	


	




if __name__ == '__main__':
	main(drug = "Pembro", mode = 'tpm_n')