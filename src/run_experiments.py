import numpy as np 
from numpy.linalg import matrix_rank, svd


import pandas as pd 
import sys 

from lifelines.datasets import load_rossi 
from lifelines import CoxPHFitter, WeibullAFTFitter,LogLogisticAFTFitter, LogNormalAFTFitter

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.model_selection import ShuffleSplit


from lifelines.utils import concordance_index
from utils import *


from metric_learn import MMC_Supervised, ITML_Supervised

import matplotlib.pyplot as plt 
import seaborn as sns 

def main():

	task_data = np.load("./data/preprocessed/Pembro/Pembro_STAD_CS.npz")


	n_splits = 10
	train_size = 0.8
	variance_pct = 0.90
	eig_thresh = 10e-8
	
	X = task_data['X'] 
	y_dst = task_data['Y_s']
	y_src = task_data['Y_c']
	A = task_data['adj_mat']
	D = np.diag(np.sum(A,axis=1))
	L = D-A 
	n_modes = 20

	vals, vectors = np.linalg.eigh(L)
	GFT = vectors[:,-n_modes:]

	plt.scatter(np.arange(vals.shape[0]),vals)
	plt.savefig("./figs/eigvals.png")
	plt.close()

	results = {'iter':[],'model':[],
		'transform':[],'test_ci':[]}

	rng = np.random.RandomState(seed=1234)
	
	skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=rng)

	for i, (train_idx,test_idx) in enumerate(skf.split(X,y_src)):
		X_train, X_test, y_train, y_test = X[train_idx,:], X[test_idx,:], y_src[train_idx],y_src[test_idx]

		scaler = StandardScaler()
		scaler = scaler.fit(X_train)

		pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct))])
		pipe_mmc  = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct)),('metric_learner',MMC_Supervised(random_state=rng))])
		pipe_mmc = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct)), ('metric_learner',ITML_Supervised(random_state=rng))])
		

		### PCA
		print("PCA")

		train_df = pd.DataFrame(pipe.fit_transform(X_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(pipe.transform(X_test))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		cox = CoxPHFitter()
		weibull = WeibullAFTFitter()
		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('pca')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('pca')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		

		### GFT 
		print("GFT")
		
		
		train_df = pd.DataFrame(X_train @ GFT)
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(X_test @ GFT)
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('gft')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('gft')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		



		### PCA + MMC
		print("PCA MMC")

		train_df = pd.DataFrame(pipe_mmc.fit_transform(X_train,y_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(pipe_mmc.transform(X_test))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		


		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('pca+mmc')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('pca+mmc')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		


		### GFT + MMC
		print("GFT MMC")
		X_train_hat =  X_train @ GFT
		X_test_hat =  X_test @ GFT


		gft_pipe = Pipeline([('scaler',StandardScaler()), ('metric_learner',ITML_Supervised(random_state=rng))])



		train_df = pd.DataFrame(gft_pipe.fit_transform(X_train_hat,y_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(gft_pipe.transform(X_test_hat))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('gft+mmc')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('gft+mmc')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)



	results = pd.DataFrame(results)
	results.to_csv("./data/initial_res.csv")
	results = {'iter':[],'model':[],
		'transform':[],'test_ci':[]}
	splitter = ShuffleSplit(n_splits=5, train_size=train_size, random_state=rng)
	for i, (train_index, test_index) in enumerate(splitter.split(X)):
		X_train, X_test, y_train, y_test = X[train_idx,:], X[test_idx,:], y_src[train_idx],y_src[test_idx]

		scaler = StandardScaler()
		scaler = scaler.fit(X_train)

		pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct))])
		pipe_mmc  = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct)),('metric_learner',MMC_Supervised(random_state=rng))])
		pipe_mmc = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = variance_pct)), ('metric_learner',ITML_Supervised(random_state=rng))])
		

		### PCA
		print("PCA")

		train_df = pd.DataFrame(pipe.fit_transform(X_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(pipe.transform(X_test))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		cox = CoxPHFitter()
		weibull = WeibullAFTFitter()
		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('pca')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('pca')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		

		### GFT 
		print("GFT")
		
		
		train_df = pd.DataFrame(X_train @ GFT)
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(X_test @ GFT)
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('gft')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('gft')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		



		### PCA + MMC
		print("PCA MMC")

		train_df = pd.DataFrame(pipe_mmc.fit_transform(X_train,y_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(pipe_mmc.transform(X_test))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		


		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('pca+mmc')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('pca+mmc')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)
		


		### GFT + MMC
		print("GFT MMC")
		X_train_hat =  X_train @ GFT
		X_test_hat =  X_test @ GFT


		gft_pipe = Pipeline([('scaler',StandardScaler()), ('metric_learner',ITML_Supervised(random_state=rng))])



		train_df = pd.DataFrame(gft_pipe.fit_transform(X_train_hat,y_train))
		train_df['duration'] = y_dst[train_idx,0]
		train_df['event'] = y_dst[train_idx,1]
		train_df['event'] = train_df['event'].astype('int64')


		test_df = pd.DataFrame(gft_pipe.transform(X_test_hat))
		test_df['duration']=y_dst[test_idx,0]
		test_df['event']=y_dst[test_idx,1]
		test_df['event']= test_df['event'].astype('int64')
		

		

		cox.fit(train_df,duration_col='duration',event_col='event')
		weibull.fit(train_df,duration_col='duration',event_col='event')
		

		test_CI = concordance_index(test_df['duration'],cox.predict_partial_hazard(test_df),test_df['event'])
		results['iter'].append(i)
		results['model'].append('cox')
		results['transform'].append('gft+mmc')
		results['test_ci'].append(test_CI)
		

		
		test_CI = concordance_index(test_df['duration'],weibull.predict_median(test_df),test_df['event'])
		results['iter'].append(i)
		results['transform'].append('gft+mmc')
		results['model'].append('weibull')
		results['test_ci'].append(test_CI)



	results = pd.DataFrame(results)
	results.to_csv("./data/initial_res_alt.csv")
	



		
		

if __name__ == '__main__':
	main()