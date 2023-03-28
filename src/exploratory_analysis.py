import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sys 
import os 
import lifelines


Rx = ['Atezo','Ipi','Nivo','Pembro']

def emap(score:float)->str:
	if np.isnan(score):
		return(np.nan)

	elif int(score)==1:
		return('observed')
	else:
		return('not observed')



def plot_swarm(df:pd.DataFrame,title:str,fname:str)->None:
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


def plot_survival_curves(df:pd.DataFrame,title:str,fname:str)->None:
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

for itx in Rx:
	df = pd.read_csv("./data/summaries/{r}.csv".format(r=itx),index_col=0)
	df['event'] = df['PFS_e'].apply(emap)
	drug_path = "./figs/{rx}/".format(rx=itx)
	# print(drug_path)
	os.makedirs(drug_path,exist_ok=True)
	plot_swarm(df,title="{rx} All Tissues".format(rx=str(itx)), fname = "./figs/{rx}/{rx}_All.png".format(rx=str(itx)))
	plot_survival_curves(df,title="{rx} All Tissues Survival".format(rx=str(itx)),fname = "./figs/{rx}/{rx}_All_Survival.png".format(rx=str(itx)))
	
	

	
	for tissue in list(pd.unique(df['TCGA_Tissue'])):
		tissue_df = df[df['TCGA_Tissue']==tissue]
		tissue_df = tissue_df.dropna(subset=['PFS_e','PFS_d'])
		if tissue_df.shape[0]>0:
			plot_swarm(tissue_df,title="{rx} {t}".format(rx=itx,t=tissue),fname = "./figs/{rx}/{rx}_{t}.png".format(rx=itx,t=tissue))
			plot_survival_curves(df,title="{rx} {t} Survival".format(rx=str(itx),t=tissue),fname = "./figs/{rx}/{rx}_{t}_Survival.png".format(rx=str(itx),t=tissue))


itx_samples = pd.read_csv("./data/raw/cri/iatlas-ici-sample_info.tsv",sep="\t")
itx_samples = itx_samples[itx_samples['Sample_Treated']==False]
itx_samples['event'] = itx_samples['PFS_e'].apply(emap)
combos = ['Ipi + Pembro','Ipi + Nivo']
itx_samples = itx_samples[itx_samples['ICI_Rx'].isin(combos)]

for combo in combos: 
	combo_df = itx_samples[itx_samples['ICI_Rx']==combo]
	elements = combo.split()
	combo_string = "".join([elements[0],"_",elements[2]])
	os.makedirs("./figs/{cs}".format(cs=combo_string),exist_ok=True)
	plot_swarm(combo_df,title="{c} All Tissue".format(c=combo_string),fname="./figs/{c}/{c}_All.png".format(c=combo_string))
	plot_survival_curves(df,title="{c} All Tissues Survival".format(c=combo_string),fname = "./figs/{c}/{c}_All_Survival.png".format(c=combo_string))

	for tissue in list(pd.unique(combo_df['TCGA_Tissue'])):
		tissue_df = combo_df[combo_df['TCGA_Tissue']==tissue]
		plot_swarm(tissue_df,title="{c} {t}".format(c=combo_string,t=tissue),fname="./figs/{c}/{c}_{t}.png".format(c=combo_string,t=tissue))
		plot_survival_curves(df,title="{c} {t} Survival".format(c=combo_string,t=tissue),fname = "./figs/{c}/{c}_{t}_Survival.png".format(c=combo_string,t=tissue))



####
#
#	value count information
#
###




CRI_DATA = "./data/raw/cri/iatlas-ici-sample_info.tsv"
sample_df = pd.read_csv(CRI_DATA,sep="\t")
sample_df = sample_df.dropna(subset = ['Response'])

# subset the samples that weren't treated at time of selection 
sample_df = sample_df[sample_df['Sample_Treated']==False]
sample_df = sample_df[['Run_ID', 'TCGA_Tissue', 'Response', 'OS_e', 'OS_d', 'PFS_e', 'PFS_d','ICI_Rx']]


responses = list(pd.unique(sample_df['Response']))
tissues = list(pd.unique(sample_df['TCGA_Tissue']))


base_tissue_counts = {t:[] for t in tissues}
base_response_counts = {r:[] for r in responses}

monotherapies = ['Atezo','Pembro','Nivo','Ipi']

rows = {'Therapy':[]}
rows.update(base_tissue_counts)
rows.update(base_response_counts)

tissue_zip = []
for i in tissues:
	for j in responses:
		tissue_zip.append("{t}_{r}".format(t=i,r=j))

for z in tissue_zip:
	rows[z]=[]



for mono in monotherapies:
	rows['Therapy'].append(mono)
	mono_df = sample_df[sample_df['ICI_Rx']==mono]
	mono_df.to_csv("./data/summaries/{drug}.csv".format(drug=mono))

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
counts_dataframe.to_csv("./data/summaries/ITx_counts.csv")
