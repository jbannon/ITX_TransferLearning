import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys 
from sklearn.pipeline import Pipeline
from sklearn.metrics import *



combo_data = np.load("../data/preprocessed/Ipi + Pembro/ALL_NMA.npz")
ipi_data  = np.load("../data/preprocessed/Ipi/SKCM_NMA.npz")
pembro_data = np.load("../data/preprocessed/Pembro/SKCM_NMA.npz")



X_ipi, y_ipi = ipi_data['X'], ipi_data['y']
X_pembro, y_pembro = pembro_data['X'], pembro_data['y']

X = combo_data['X']
y = combo_data['y']


scaler = StandardScaler()
logistic = LogisticRegression(solver='saga',max_iter = 10000,tol = 1e-3)
pca = PCA()
loo_splitter = LeaveOneOut()

pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])



# SVC_param_grid = {'C': [0.01, 0.001, 0.0001,0.1, 1, 10, 100, 1000],'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}

LR_param_grid = {
    "pca__n_components": [2,5,0.9],
    "logistic__C": np.logspace(-5, 4),
    "logistic__penalty":['l1','l2']
}


pred_responses = []
predicted_prob = []
true_responses = []
fold = []
transform = []

i = 1 

pembro_cv = GridSearchCV(pipe,param_grid=LR_param_grid,scoring='roc_auc', cv=5, n_jobs=5).fit(X_pembro, y_pembro)
pembro_model = pembro_cv.best_estimator_

ipi_cv = GridSearchCV(pipe,param_grid=LR_param_grid,scoring='roc_auc', cv=5, n_jobs=5).fit(X_ipi, y_ipi)
ipi_model = ipi_cv.best_estimator_

for trn_idx,tst_idx in loo_splitter.split(X):
	X_train, X_test, y_train, y_test = X[trn_idx,:],X[tst_idx,:], y[trn_idx],y[tst_idx]
	

	lr_cv = GridSearchCV(pipe,param_grid=LR_param_grid,scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)
	pred_responses.append(lr_cv.best_estimator_.predict(X_test)[0])
	predicted_prob.apped(lr_cv.best_estimator_.predict(X_test)[0][1])
	true_responses.append(y_test[0])
	fold.append(i)
	transform.append('none')

	Xp = pembro_model.predict_proba(X_train)
	Xi = ipi_model.predict_proba(X_train)
	Xcomb_train = np.hstack((Xp,Xi))

	Xp = pembro_model.predict_proba(X_test)
	Xi = ipi_model.predict_proba(X_test)
	Xcomb_test = np.hstack((Xp,Xi))
	

	lr_cv = GridSearchCV(pipe,param_grid=LR_param_grid,scoring='roc_auc', cv=5, n_jobs=5).fit(Xcomb_train, y_train)
	pred_responses.append(lr_cv.best_estimator_.predict(Xcomb_test)[0])
	predicted_prob.apped(lr_cv.best_estimator_.predict(X_test)[0][1])
	true_responses.append(y_test[0])
	fold.append(i)
	transform.append('mono')


	i=i+1



res = pd.DataFrame({'pred':pred_responses,'actual':true_responses,'fold':fold,'transform':transform})
res.to_csv("../data/results/baseline_lr.csv")