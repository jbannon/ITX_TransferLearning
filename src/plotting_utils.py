import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

def plot_pca(
	X:np.array,
	y:np.array,
	title:str = None,
	file_path:str = None
	)->None:
	pipe = Pipeline([ ("scaler",StandardScaler()),("DimRed",PCA(n_components=2))])
	X_embedded = pipe.fit_transform(X)
	df = pd.DataFrame({'pc1':X_embedded[:, 0],'pc2':X_embedded[:, 1],'class':y})
	
	sns.scatterplot(df,x='pc1',y='pc2', hue='class')
	

	if title is not None:
		plt.title(title)
	
	if file_path is not None:
		plt.savefig(file_path)
	else:
		plt.show()

	plt.close()


	

def plot_umap(X:np.array,
	y:np.array,
	mindist:float = 0.1,
	n_nbrs:int = 5,
	metric:str = 'euclidean',
	title:str = None,
	file_path:str = None
	)->None:

	um = umap.UMAP(min_dist=mindist,n_neighbors=n_nbrs,metric=metric)
	X_embedded = um.fit_transform(X)
	df = pd.DataFrame({'umap1':X_embedded[:, 0],'umap2':X_embedded[:, 1],'class':y})
	sns.scatterplot(df,x='umap1',y='umap2', hue='class')
	if title is not None:
		plt.title(title)
	
	if file_path is not None:
		plt.savefig(file_path)
	else:
		plt.show()

	plt.close()

def plot_tsne(
	X:np.array,
	y:np.array,
	per:int = 5,
	title:str = None,
	file_path:str = None
	)->None:

	if per>X.shape[0]:
		return
	
	tsne = TSNE(perplexity = per)
	X_embedded = tsne.fit_transform(X)
	df = pd.DataFrame({'tsne1':X_embedded[:, 0],'tsne2':X_embedded[:, 1],'class':y})

	sns.scatterplot(df,x='tsne1',y='tsne2', hue='class')
	if title is not None:
		plt.title(title)
	
	if file_path is not None:
		plt.savefig(file_path)
	else:
		plt.show()

	plt.close()