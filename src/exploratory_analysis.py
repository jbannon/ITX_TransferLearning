import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sys 
import os 
import lifelines




def emap(score:float)->str:
	if np.isnan(score):
		return(np.nan)

	elif int(score)==1:
		return('observed')
	else:
		return('not observed')


def plot_cat_barchart(
	df:pd.DataFrame,
	title:str = None,
	fname:str = None
	)->None:
	cat_plot = sns.catplot(data=df, x='response', y='sample count', kind='bar', hue='Tissue', legend=True,aspect=2)
	ax = cat_plot.facet_axis(0, 0)
	for c in ax.containers:
		labels = [str(int(v.get_height())) for v in c]
		ax.bar_label(c, labels=labels, label_type='edge')

	cat_plot.set(title=title,xlabel = "Response",ylabel = "Sample Counts")
	plt.savefig(fname,bbox_inches="tight")
	plt.close()


def plot_swarm(
	df:pd.DataFrame,
	title:str,
	fname:str
	)->None:

	ORDER = ['Progressive Disease','Stable Disease','Partial Response','Complete Response']
	ORDER.reverse()
	ax = sns.swarmplot(data=df, x="PFS_d", y="Response", hue="event",order=ORDER)
	ax.set(ylabel="")
	ax.set(xlabel="Progression Free Survival Days")
	plt.title(title)
	plt.legend(title="Progression",loc='lower right')
	plt.tight_layout()
	plt.savefig(fname)
	plt.close()


def plot_survival_curves(
	df:pd.DataFrame,
	title:str,fname:str
	)->None:
	df = df.dropna(subset=['PFS_e'])
	responses = list(pd.unique(df['Response']))
	kmf = lifelines.KaplanMeierFitter()
	ax = plt.subplot(111)
	
	for res in responses:
		
		subset = df[df["Response"] == res]
		kmf.fit(durations=subset['PFS_d'], 
        	event_observed=subset['PFS_e'], 
        	label=res)
		kmf.plot(ax=ax,show_censors=True, ci_show=False)
		
	plt.title(title)
	plt.savefig(fname)
	plt.close()


CRI_DATA = "../data/raw/cri/iatlas-ici-sample_info.tsv"
sample_df = pd.read_csv(CRI_DATA,sep="\t")
sample_df = sample_df.dropna(subset = ['Response'])



sample_df = sample_df[sample_df['Sample_Treated']==False] # subset the samples that weren't treated at time of selection; treated samples will confound 


sample_df = sample_df[['Run_ID', 'TCGA_Tissue', 'Response', 'OS_e', 'OS_d', 'PFS_e', 'PFS_d','ICI_Rx']]
sample_df['Binarized_Response'] = sample_df['Response'].apply(lambda x:"Responder" if x in ['Complete Response', 'Partial Response'] else "Non-Responder")

responses = list(pd.unique(sample_df['Response']))
binary_response = list(pd.unique(sample_df['Binarized_Response']))

tissues = list(pd.unique(sample_df['TCGA_Tissue']))


base_tissue_counts = {t:[] for t in tissues}
base_response_counts = {r:[] for r in responses}

monoTherapies = ['Atezo','Pembro','Nivo','Ipi']
comboTherapies = ['Ipi + Pembro','Ipi + Nivo']

rows = {'Therapy':[]}
rows.update(base_tissue_counts)
rows.update(base_response_counts)

tissue_zip = []


for i in tissues:
	for j in responses:
		tissue_zip.append("{t}_{r}".format(t=i,r=j))

for z in tissue_zip:
	rows[z]=[]

for mono in monoTherapies:
	rows['Therapy'].append(mono)
	mono_df = sample_df[sample_df['ICI_Rx']==mono]
	mono_df.to_csv("../data/summaries/{drug}.csv".format(drug=mono))

	tissue_counts = mono_df['TCGA_Tissue'].value_counts().to_dict()
	response_counts = mono_df['Response'].value_counts().to_dict()
	
	
	

	
	


	for tissue in tissues:
		if tissue in tissue_counts.keys():
			rows[tissue].append(tissue_counts[tissue])
		else:
			rows[tissue].append(0)
	
	
	for response in responses:
		if response in response_counts.keys():
			rows[response].append(response_counts[response])
		else:rows[response].append(0)


	temp = {}

	for t in list(tissue_counts.keys()):

		tissue_df = mono_df[mono_df['TCGA_Tissue']==t]
		tissue_response_counts = tissue_df['Response'].value_counts().to_dict()
		tissue_response_counts = {"{ts}_{k}".format(ts=t,k=key):tissue_response_counts[key] for key in tissue_response_counts.keys()}
		temp.update(tissue_response_counts)
		
	for tz in tissue_zip:
		if tz in temp.keys():
			rows[tz].append(temp[tz])
		else:
			rows[tz].append(0)
	

					
		

counts_dataframe = pd.DataFrame(rows)
# print(counts_dataframe)
counts_dataframe.to_csv("../data/summaries/ITx_counts.csv")


Binary_Responses = ['Non-Responder','Responder']
Responses = ['Progressive Disease','Stable Disease','Partial Response','Complete Response']

for itx in monoTherapies:
	df = pd.read_csv("../data/summaries/{r}.csv".format(r=itx),index_col=0)
	df['event'] = df['PFS_e'].apply(emap)
	drug_path = "../figs/{rx}/".format(rx=itx)
	# print(drug_path)
	os.makedirs(drug_path,exist_ok=True)
	plot_swarm(df,title="{rx} All Tissues".format(rx=str(itx)), fname = "../figs/{rx}/{rx}_All.png".format(rx=str(itx)))
	plot_survival_curves(df,title="{rx} All Tissues Survival".format(rx=str(itx)),fname = "../figs/{rx}/{rx}_All_Survival.png".format(rx=str(itx)))
	
	response_counts = df['Response'].value_counts().to_dict()
	binary_response_counts = df['Binarized_Response'].value_counts().to_dict()

	tissue_labels = []
	binary_tissue_labels = []

	response_labels = []
	binary_response_labels = []
	
	rcounts = []
	brcounts = []

	for r in Responses:
		tissue_labels.append('All Tissues')
		response_labels.append(r)
		if r in response_counts.keys():
			rcounts.append(response_counts[r])
		else:
			rcounts.append(0)

	for br in Binary_Responses:
		binary_tissue_labels.append("All Tissue")
		binary_response_labels.append(br)
		if br in binary_response_counts.keys():
			brcounts.append(binary_response_counts[br])
		else:
			brcounts.append(0)



	

	
	for tissue in list(pd.unique(df['TCGA_Tissue'])):
		tissue_df = df[df['TCGA_Tissue']==tissue]
		tissue_classificaction = tissue_df.copy()

		tissue_df = tissue_df.dropna(subset=['PFS_e','PFS_d'])
		if tissue_df.shape[0]>0:
			plot_swarm(tissue_df,title="{rx} {t}".format(rx=itx,t=tissue),fname = "../figs/{rx}/{rx}_{t}.png".format(rx=itx,t=tissue))
			plot_survival_curves(df,title="{rx} {t} Survival".format(rx=str(itx),t=tissue),fname = "../figs/{rx}/{rx}_{t}_Survival.png".format(rx=str(itx),t=tissue))

		tissue_response_counts = tissue_classificaction['Response'].value_counts().to_dict()
		tissue_binary_response_counts = tissue_classificaction['Binarized_Response'].value_counts().to_dict()

		for r in Responses:
			tissue_labels.append(tissue)
			response_labels.append(r)
			if r in tissue_response_counts.keys():
				rcounts.append(tissue_response_counts[r])
			else:
				rcounts.append(0)

		for br in Binary_Responses:
			binary_tissue_labels.append(tissue)
			binary_response_labels.append(br)
			if br in tissue_binary_response_counts.keys():
				brcounts.append(tissue_binary_response_counts[br])
			else:
				brcounts.append(0)

	catplot_df = pd.DataFrame({'Tissue':tissue_labels,'response':response_labels,'sample count':rcounts})

	plot_cat_barchart(catplot_df,title = "{drug} Reponse Counts".format(drug=itx),fname = "../figs/{rx}/{rx}_response_breakdown.png".format(rx=str(itx)))
	
	catplot_df = pd.DataFrame({'Tissue':binary_tissue_labels,'response':binary_response_labels,'sample count':brcounts})
	plot_cat_barchart(catplot_df,title = "{drug} Binary Reponse Counts".format(drug=itx),fname = "../figs/{rx}/{rx}_binary_response_breakdown.png".format(rx=str(itx)))




itx_samples = pd.read_csv("../data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")
itx_samples['Binarized_Response'] = itx_samples['Response'].apply(lambda x:"Responder" if x in ['Complete Response','Partial Response'] else "Non-Responder")
itx_samples = itx_samples[itx_samples['Sample_Treated']==False]
itx_samples['event'] = itx_samples['PFS_e'].apply(emap)

itx_samples = itx_samples[itx_samples['ICI_Rx'].isin(comboTherapies)]

for combo in comboTherapies: 
	combo_df = itx_samples[itx_samples['ICI_Rx']==combo]
	elements = combo.split()
	combo_string = "".join([elements[0],"_",elements[2]])
	os.makedirs("../figs/{cs}".format(cs=combo_string),exist_ok=True)
	plot_swarm(combo_df,title="{c} All Tissue".format(c=combo_string),fname="../figs/{c}/{c}_All.png".format(c=combo_string))
	plot_survival_curves(df,title="{c} All Tissues Survival".format(c=combo_string),fname = "../figs/{c}/{c}_All_Survival.png".format(c=combo_string))


	response_counts = combo_df['Response'].value_counts().to_dict()
	binary_response_counts = combo_df['Binarized_Response'].value_counts().to_dict()

	tissue_labels = []
	binary_tissue_labels = []

	response_labels = []
	binary_response_labels = []
	
	rcounts = []
	brcounts = []

	for r in Responses:
		tissue_labels.append('All Tissues')
		response_labels.append(r)
		if r in response_counts.keys():
			rcounts.append(response_counts[r])
		else:
			rcounts.append(0)

	for br in Binary_Responses:
		binary_tissue_labels.append("All Tissue")
		binary_response_labels.append(br)
		if br in binary_response_counts.keys():
			brcounts.append(binary_response_counts[br])
		else:
			brcounts.append(0)


	for tissue in list(pd.unique(combo_df['TCGA_Tissue'])):
		tissue_df = combo_df[combo_df['TCGA_Tissue']==tissue]
		plot_swarm(tissue_df,title="{c} {t}".format(c=combo_string,t=tissue),fname="../figs/{c}/{c}_{t}.png".format(c=combo_string,t=tissue))
		plot_survival_curves(df,title="{c} {t} Survival".format(c=combo_string,t=tissue),fname = "../figs/{c}/{c}_{t}_Survival.png".format(c=combo_string,t=tissue))


		tissue_response_counts = tissue_df['Response'].value_counts().to_dict()
		tissue_binary_response_counts = tissue_df['Binarized_Response'].value_counts().to_dict()

		for r in Responses:
			tissue_labels.append(tissue)
			response_labels.append(r)
			if r in tissue_response_counts.keys():
				rcounts.append(tissue_response_counts[r])
			else:
				rcounts.append(0)

		for br in Binary_Responses:
			binary_tissue_labels.append(tissue)
			binary_response_labels.append(br)
			if br in tissue_binary_response_counts.keys():
				brcounts.append(tissue_binary_response_counts[br])
			else:
				brcounts.append(0)

	catplot_df = pd.DataFrame({'Tissue':tissue_labels,'response':response_labels,'sample count':rcounts})
	plot_cat_barchart(catplot_df,title = "{drug} Reponse Counts".format(drug=itx),fname = "../figs/{rx}/{rx}_response_breakdown.png".format(rx=combo_string))
	
	catplot_df = pd.DataFrame({'Tissue':binary_tissue_labels,'response':binary_response_labels,'sample count':brcounts})
	plot_cat_barchart(catplot_df,title = "{drug} Binary Reponse Counts".format(drug=itx),fname = "../figs/{rx}/{rx}_binary_response_breakdown.png".format(rx=combo_string))
	
	





