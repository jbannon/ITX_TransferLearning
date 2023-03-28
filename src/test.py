import pandas as pd 
import numpy as np 
from lifelines.datasets import load_rossi 
from lifelines import CoxPHFitter, WeibullAFTFitter,LogLogisticAFTFitter, LogNormalAFTFitter

from sklearn.feature_selection import VarianceThreshold 
from metric_learn import NCA


rossi = load_rossi()
cph = CoxPHFitter()
cph.fit(rossi, duration_col='week', event_col='arrest')

cph.print_summary()
print(rossi)