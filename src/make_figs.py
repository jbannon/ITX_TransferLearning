import seaborn as sns 
import pandas as pd  
import matplotlib.pyplot as plt



results = pd.read_csv("./data/initial_res.csv",index_col=0)

for model in list(pd.unique(results['model'])):
	temp = results[results['model']==model]
	sns.violinplot(temp,x="transform",y='test_ci')
	plt.title("{m} test concordance".format(m=model))
	plt.savefig("./figs/{m}_initial_violin.png".format(m=model))
	plt.close()



	sns.swarmplot(data=temp, x="transform", y="test_ci")
	plt.title("{m} test concordance".format(m=model))
	plt.savefig("./figs/{m}_initial_swarm.png".format(m=model))
	plt.close()



results = pd.read_csv("./data/initial_res_alt.csv",index_col=0)

for model in list(pd.unique(results['model'])):
	temp = results[results['model']==model]
	sns.violinplot(temp,x="transform",y='test_ci')
	plt.title("{m} test concordance".format(m=model))
	plt.savefig("./figs/{m}_initial_alt_split_violin.png".format(m=model))
	plt.close()



	sns.swarmplot(data=temp, x="transform", y="test_ci")
	plt.title("{m} test concordance".format(m=model))
	plt.savefig("./figs/{m}_initial_alt_split_swarm.png".format(m=model))
	plt.close()



