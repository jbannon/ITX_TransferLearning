import sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


Rx_2_Transfer_Pairs = {
	'Atezo':[("KIRC","BLCA"), ("BLCA","KIRC")],
	'Pembro':[("SKCM","STAD")],
	'Nivo':[("SKCM","KIRC")]}

code_2_type = {1:'Responder',0:'Nonresponder'}

Preprocessing = {'NMA':'NMA','prenorm':'Quantile Normalized'}


def make_corr_matrix(
	X1:np.array,
	X2:np.array
	)->np.array:
	assert X1.shape[1]==X2.shape[1]
	result = np.empty( (X1.shape[0],X2.shape[0]))
	for i in range(X1.shape[0]):
		for j in range(X2.shape[0]):
			result[i,j] = np.dot(X1[i,:],X2[j,:])/(np.linalg.norm(X1[i,:])*np.linalg.norm(X2[j,:]))

	return result

for treatment in Rx_2_Transfer_Pairs.keys():
	print("Working on {t}".format(t=treatment))
	for pairs in Rx_2_Transfer_Pairs[treatment]:
		src_tissue = pairs[0]
		tgt_tissue = pairs[1]

		for process_type in Preprocessing.keys():
			src_task = np.load("../data/preprocessed/{rx}/{s}_{p}.npz".format(rx=treatment,s=src_tissue,p=process_type))
			tgt_task = np.load("../data/preprocessed/{rx}/{s}_{p}.npz".format(rx=treatment,s=tgt_tissue,p=process_type))

			X_src = src_task['X']
			y_src = src_task['y']
			X_tgt = tgt_task['X']
			y_tgt =  tgt_task['y']

			for c in code_2_type.keys():
				for c_ in code_2_type.keys():
					src_sub = np.where(y_src==c)
					tgt_sub = np.where(y_tgt==c_)
					subset_correlation = make_corr_matrix(X_src[src_sub[0],:], X_tgt[tgt_sub[0],:])
					fig = sns.heatmap(subset_correlation)
					fig.set(
						xlabel="{tgt}".format(tgt=tgt_tissue),
						ylabel="{src}".format(src=src_tissue),
						title= "{rx} {src} ({c1}) {dst} ({c2}) Correlation\n {p}".format(rx = treatment,src = src_tissue,dst = tgt_tissue,c1 = code_2_type[c], c2= code_2_type[c_], p=Preprocessing[process_type]))
					plt.savefig("../figs/{rx}/exploratory/{s}_{c1}_{t}_{c2}_{p}.png".format(rx=treatment,s=src_tissue,t=tgt_tissue,p=process_type,c1=code_2_type[c],c2=code_2_type[c_]))
					plt.close()

				

for process_type in Preprocessing.keys():
	X_pem = np.load("../data/preprocessed/Pembro/SKCM_{p}.npz".format(p=process_type))
	Xp = X_pem['X']
	yp = X_pem['y']
	X_ipi = np.load("../data/preprocessed/Pembro/SKCM_{p}.npz".format(p=process_type))
	Xi = X_ipi['X']
	yi = X_ipi['y']
	X_tgt = np.load("../data/preprocessed/Ipi + Pembro/ALL_{p}.npz".format(p=process_type))
	X_t = X_tgt['X']
	y_t = X_tgt['y']
	X_t = X_t[np.where(y_t==1)[0],:]
	for c1 in code_2_type.keys():
		for c2 in code_2_type.keys():
			X1 = Xp[np.where(yp==c1)[0],:]
			print(X1.shape)
			
			X2 = Xi[np.where(yi==c2)[0],:]
			X_src = np.vstack((X1,X2))
			subset_correlation = make_corr_matrix(X_src, X_t)
			fig = sns.heatmap(subset_correlation)
			fig.set(title= "Pembro ({c1}; {r}) Ipi ({c2}) \n Combo Responders {p}".format(c1=code_2_type[c1], r=str(X1.shape[0]),c2= code_2_type[c2], p=Preprocessing[process_type]))
			plt.savefig("../figs/{rx}/exploratory/{c1}_{c2}_{p}.png".format(rx="Ipi + Pembro",c1=code_2_type[c1],c2=code_2_type[c2],p=Preprocessing[process_type]))
			plt.close()


	



