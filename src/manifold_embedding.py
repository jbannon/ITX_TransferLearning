import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import seaborn as sns
import utils
from typing import List, Tuple, Dict
import sys
from metric_learn import MMC_Supervised, ITML_Supervised, LMNN, NCA 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import umap

import matplotlib.pyplot as plt 
import seaborn as sns 


def plot_pca(
	X:np.array,
	y:np.array,
	title:str = None
	)->None:
	pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=2))])
	X_embedded = pipe.fit_transform(X)
	df = pd.DataFrame({'pc1':X_embedded[:, 0],'pc2':X_embedded[:, 1],'class':y})
	sns.scatterplot(df,x='pc1',y='pc2', hue='class')
	plt.show()


	

def plot_umap(X:np.array,
	y:np.array,
	title:str = None):
	um = umap.UMAP()
	X_embedded = um.fit_transform(X)
	df = pd.DataFrame({'umap1':X_embedded[:, 0],'umap2':X_embedded[:, 1],'class':y})
	sns.scatterplot(df,x='umap1',y='umap2', hue='class')
	plt.show()

def plot_tsne(
	X:np.array,
	y:np.array,
	title:str = None):
	per = 50
	tsne = TSNE(perplexity = per)
	X_embedded = tsne.fit_transform(X)
	df = pd.DataFrame({'tsne1':X_embedded[:, 0],'tsne2':X_embedded[:, 1],'class':y})

	sns.scatterplot(df,x='tsne1',y='tsne2', hue='class')
	if title is not None:
		plt.title(title + " " + str(per))
	plt.show()	


def plot_metric_learning(
	X:np.array,
	y:np.array
	)->None:

	print("fitting Met Learn")
	NoML_Pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=50))])
	# MMC_Pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=10)),("MetLearner",MMC_Supervised())])
	ITML_Pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=0.9)),("MetLearner",ITML_Supervised())])
	LMNN_Pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=0.9)),("MetLearner",LMNN())])
	NCA_Pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=0.9)),("MetLearner",NCA())])
	PipeDict = {"No MetLrn": NoML_Pipe,"ITML":ITML_Pipe,"LMNN":LMNN_Pipe,"NCA":NCA_Pipe}
	#"MMC":MMC_Pipe,
	for pipe_name in PipeDict.keys():
		print(pipe_name)
		pipe = PipeDict[pipe_name]
		X = pipe.fit_transform(X,y)
		# plot_tsne(X,y,title=pipe_name)
		plot_umap(X,y,title=pipe_name)
		plot_pca(X,y,title=pipe_name)
	sys.exit(1)





def plot_responder_correlation_heatmaps():
	pass

def main(): 

	# GM = utils.collect_GeneMANIA_genes()
	# PC = utils.collect_PathwayCommons_genes()
	# S700 = utils.collect_STRINGdb_genes()
	# S900 = utils.collect_STRINGdb_genes(score_threshold=900)
	# S500 = utils.collect_STRINGdb_genes(score_threshold=500)

	np.random.seed(1234)



	Treatments = ['Atezo','Pembro','Nivo','Ipi','Ipi + Pembro','Ipi + Nivo']
	hallmark_genes = utils.get_hallmark_genes("../data/raw/mdsig_hallmarks.txt")
	median_expression = utils.fetch_normal_expression_medians("../data/raw/GTEx_Medians.gct")
	# GeneSets = {'GeneMANIA':GM,'PathwayCommons':PC,'SDB_500':S500, 'SDB_700':S700,'SDB_900':S900,'Hallmark':hallmark_genes}
	GeneSets = {'Hallmark':hallmark_genes,'All':[]}
	

	# sample_tpm = pd.read_csv('../data/raw/cri/iatlas-ici-genes_norm.tsv', sep="\t")
	
	sample_norm = pd.read_csv("../data/raw/cri/iatlas-ici-genes_norm.tsv", sep="\t")




	sample_info = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")
	sample_info['BinarizedResponse'] = sample_info['Response'].apply(lambda x: "R" if x in ['Complete Response','Partial Response'] else "NR")
	sample_info['Response_Class'] = pd.Categorical(sample_info['BinarizedResponse'])
	sample_info['Response_Class'] = sample_info['Response_Class'].cat.codes
	sample_info = sample_info.dropna(subset=['Response'])

	sample_info = sample_info[sample_info['Sample_Treated']==False]
	sample_info = sample_info[['Run_ID','TCGA_Tissue','BinarizedResponse','ICI_Rx','Response_Class']]
	 
	# tpm_data = sample_tpm.merge(sample_info,left_on="sample",right_on="Run_ID")
	normalized_data = sample_norm.merge(sample_info,left_on="sample",right_on="Run_ID")

	for Rx in Treatments:
		treated_samples = normalized_data[normalized_data['ICI_Rx']==Rx]
		y = treated_samples['Response_Class'].to_numpy()
		expression_info = treated_samples.drop(columns=['sample','Run_ID','TCGA_Tissue','BinarizedResponse','ICI_Rx','Response_Class'])
		for geneset in GeneSets.keys():
			if geneset=='All':
				network_genes = list(expression_info.columns)
			else:
				network_genes = GeneSets[geneset]
			common_genes = utils.intersect_gene_sets(gene_sets=[network_genes,list(expression_info.columns)])
			X = expression_info[common_genes].to_numpy()

			plot_metric_learning(X,y)

		sys.exit(1)





		

if __name__ == '__main__':
	main()


