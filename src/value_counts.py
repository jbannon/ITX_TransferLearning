import pandas as pd
import sys 




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
	