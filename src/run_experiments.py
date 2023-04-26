import sys 
import os 
import argparse
import yaml 
from typing import Union, List, Dict

import numpy as np 
from numpy.linalg import matrix_rank, svd

import pandas as pd 

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.manifold import TSNE
from sklearn.model_selection import ShuffleSplit


from metric_learn import MMC_Supervised, ITML_Supervised, LMNN, NCA 






"""

TODO: create the dataset with the drugs on 

"""



	
def main(
	config:Dict
	):
	print(config)
	

	
	

	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-c",help="The config file for these experiments")
	args = parser.parse_args()
	


	with open(args.c) as file:
		config = yaml.safe_load(file)
	

	main(config)