perplexity_range = np.arange(5,100,15)
min_distance_range = [0.8]


os.makedirs(fig_dir+"ALL/", exist_ok = True)
	metrics = ['euclidean','correlation']

	for emb in ['UMAP','PCA','TSNE']:
		os.makedirs(fig_dir+"ALL/{e}/".format(e=emb), exist_ok = True)
		if emb =="UMAP":
			for met in metrics:
				os.makedirs(fig_dir+"ALL/{e}/{m}/".format(e=emb,m=met), exist_ok = True)

	print("plotting umap visualizations")
	n_nbr_range = np.arange(5,int(1.0*rx_NMA_data.shape[0]/5),5)
	for md in min_distance_range:
		for nbrs in n_nbr_range:
			for met in metrics:
				plotting_utils.plot_umap(
					X=rx_NMA_data,
					y=response,
					mindist=md,
					n_nbrs = nbrs,
					metric = met,
					title = "{r} All Tissues (Normal Median Scaled)".format(r=rx,),
					file_path = fig_dir+"ALL/{e}/{m}/{d}_{k}_NMA.png".format(e='UMAP',d=md,k=nbrs,m=met)
					)
				plotting_utils.plot_umap(
					X=rx_norm_data,
					y=response,
					mindist=md, 
					n_nbrs = nbrs,
					metric = met,
					title = "{r} All Tissues (Prenormalized)".format(r=rx),
					file_path = fig_dir+"ALL/{e}/{m}/{d}_{k}_prenorm.png".format(e='UMAP',d=md,k=nbrs,m=met)
					)

	print("plotting pca visualizations")
	plotting_utils.plot_pca(
		X=rx_NMA_data,
		y=response,
		title = "{r} All Tissues (Normal Median Scaled)".format(r=rx),
		file_path = fig_dir+"ALL/{e}/PCA_NMA.png".format(e='PCA')
		)
	plotting_utils.plot_pca(
		X=rx_norm_data,
		y=response,
		title = "{r} All Tissues (Prenormalized)".format(r=rx),
		file_path = fig_dir+"ALL/{e}/PCA_prenorm.png".format(e='PCA')
		)

	print("plotting tsne visualizations")
	
	for per in perplexity_range:
		
		plotting_utils.plot_tsne(
		X=rx_NMA_data,
		y=response,
		title = "{r} All Tissues (Normal Median Scaled)".format(r=rx),
		file_path = fig_dir+"ALL/{e}/{p}_NMA.png".format(e='TSNE',p=str(per))
		)

		plotting_utils.plot_tsne(
		X=rx_norm_data,
		y=response,
		title = "{r} All Tissues (Prenormalized)".format(r=rx),
		file_path = fig_dir+"ALL/{e}/{p}_prenorm.png".format(e='TSNE',p=str(per))
		)
			
	n_nbr_range = np.arange(5,int(1.0*tissue_NMA_data.shape[0]/5),5)
		for md in min_distance_range:
			for nbrs in n_nbr_range:
				for met in metrics:
					plotting_utils.plot_umap(
						X=tissue_NMA_data,
						y=tissue_response,
						mindist=md,
						n_nbrs = nbrs,
						metric = met,
						title = "{r} {t} (Normal Median Scaled)".format(t=tissue,r=rx),
						file_path = fig_dir+"{t}/{e}/{m}/{d}_{k}_NMA.png".format(t=tissue,e='UMAP',d=md,k=nbrs,m=met)
						)
					plotting_utils.plot_umap(
						X=tissue_norm_data,
						y=tissue_response,
						mindist=md, 
						n_nbrs = nbrs,
						metric = met,
						title = "{r} {t} (Prenormalized)".format(t=tissue,r=rx),
						file_path = fig_dir+"{t}/{e}/{m}/{d}_{k}_prenorm.png".format(t=tissue,e='UMAP',d=md,k=nbrs,m=met)
						)

		print("plotting pca visualizations")
		plotting_utils.plot_pca(
			X=tissue_NMA_data,
			y=tissue_response,
			title = "{r} {t} (Normal Median Scaled)".format(t=tissue,r=rx),
			file_path = fig_dir+"{t}/{e}/PCA_NMA.png".format(t=tissue,e='PCA')
			)
		plotting_utils.plot_pca(
			X=tissue_norm_data,
			y=tissue_response,
			title = "{r} {t} (Prenormalized)".format(t=tissue,r=rx),
			file_path = fig_dir+"{t}/{e}/PCA_prenorm.png".format(t=tissue,e='PCA')
			)

		print("plotting tsne visualizations")
		
		for per in perplexity_range:
			
			plotting_utils.plot_tsne(
			X=tissue_NMA_data,
			y=tissue_response,
			title = "{r} {t} (Normal Median Scaled)".format(r=rx,t=tissue),
			file_path = fig_dir+"{t}/{e}/{p}_NMA.png".format(t=tissue,e='TSNE',p=str(per))
			)

			plotting_utils.plot_tsne(
			X=tissue_norm_data,
			y=tissue_response,
			title = "{r} {t} (Prenormalized)".format(r=rx,t=tissue),
			file_path = fig_dir+"{t}/{e}/{p}_prenorm.png".format(t=tissue,e='TSNE',p=str(per))
			)
				

			