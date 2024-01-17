#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Fri Nov 19 11:53:47 2021

# @author: iramirezg@mondraon.edu
# """
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Nov 15 10:35:44 2021

# @author: iramirezg
# """

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sb
import sys
import time

startTime = time.time()

startTime_ns = time.time_ns()

delta_T=60 #sampling rate, mins! -> 60 mins?
# --------------------
# DATA READ
# --------------------
filename = "/home/iramirezg/Dropbox (MGEP)/Datasets/PowerDelivery/m60t1fUTC.csv" #FIRST FILE
execution_date = datetime.now()
result_dir = '' #'results/EAAI/'
data = pd.read_csv(filename,parse_dates=['DateTime'])
data.set_index(['DateTime'])
data = data.rename(columns={'water':'Moisture in top oil','Tamb':'Ambient temperature'})
target = 'TOT' 
features = list(data.keys())
#Adostutakoak kendu
features.remove('DateTime')
features.remove('TOT')
features.remove('Unnamed: 0')
#features.remove('Unnamed: 0.1')
features.remove('press')
#features.remove('water')
features.remove('moist')
features.remove('100 metre V wind component')
features.remove('Large-scale precipitation fraction')
#100 metre V wind component
#features_iec = list(['Load','Tamb']) 
#features_custom = list(['Load','Tamb','fdir']) #

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data.dropna(axis='rows',how='any',inplace=True)

dt_data = data.loc[data['DateTime'].between('2018-11-30','2019-02-03')] #BEST

dt_data.to_csv("tmp.csv",index=False)
dt_data = pd.read_csv("tmp.csv",parse_dates=['DateTime'])

result = dt_data

featuresPR = list()

LoadColumn = np.zeros(len(features))
for f in features:
    LoadColumn[features.index(f)],pp = pearsonr(dt_data['Ambient temperature'],dt_data[f])
#LoadCorrDF = pd.DataFrame(LoadCorr)
LoadCorr = pd.DataFrame(features)
LoadCorr['r'] = LoadColumn

LoadCorr = LoadCorr.sort_values('r',ascending=False)

fig,ax = plt.subplots(figsize=(22,20))
varFC = {}
#for var in selected_mix_features:
for var in LoadCorr[0]:
    varFC[var] = dt_data[var] #[x]result[var]
df = pd.DataFrame(varFC)
corr = df.corr()
lbs = list()
for col in corr.columns:
    lbs.append(col)
for f in features:
    corr[f][f] = np.nan

axhm = sb.heatmap(corr, cmap='coolwarm',yticklabels=lbs) #cbar_ax=cbar_ax)

ax_colours=['black']*len(features)
for f in features:
    if f in featuresPR:
        ax_colours[features.index(f)]='red'
#pylab.show()
[t.set_color(i) for (i,t) in zip(ax_colours,axhm.xaxis.get_ticklabels())]
#axhm.set_xticklabels(lbs,color=ax_colours)
fig.tight_layout()
plt.savefig('heatmapAU.svg',bbox_inches='tight',dpi=1200,format='svg')

#BEGIN FEATURE SELECTION
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
#from sklearn.model_selection import KFold , StratifiedKFold #,RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

EstimatorModels = [ RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor ]
train_sizes = [0.5,0.75]
param_grid = {}
param_grid[AdaBoostRegressor.__name__] = {
    'n_estimators' : [50,100,500], #[10,50,100,200,500]
    'base_estimator__min_samples_leaf' : [1,3,5,10,20,50],
    'base_estimator__max_leaf_nodes' : [None,5,10,50],
    'base_estimator__max_depth' : [None,3,10,20],
    }
param_grid[RandomForestRegressor.__name__] = {
    'n_estimators' : [50,100,200,500], #[10,50,100,200,500]
    'min_samples_leaf' : [1,3,5,10,20],
    'max_leaf_nodes' : [None, 3, 10, 20], #[None,3,5,10,20,50],
    #'max_depth' : [None,3,10,20], #[None,3,5,10,20,50]
    }
param_grid[GradientBoostingRegressor.__name__] = {
    'n_estimators' : [200,500], #[10,50,100,200,500]
    'min_samples_leaf' : [1,3,5],  #[1,3,5,10,20],
    'max_leaf_nodes' : [None,10,20], #[None,3,5,10,20,50]
    'max_depth' : [None,3,10,20], #[None,3,5,10,20,50]
    }

class FeatureSelector:
    def __init__(self,pddf,target=None,anal_feat=None):
        self.X = pddf
        self.param = None
        self.target = target
        self.features = list(self.X.keys())
        self.features.remove(self.target)
        self.anft = list()
        #self.selected_features = list()
        self.REPEAT_TIME = 3
        if anal_feat is not None:
            for feat in anal_feat:
                if feat in self.X.keys():
                    self.anft.append(feat)
                    
    def CorrelationAnalysis(self,correlationThreshold=0.9):
        featuresPR = list()
        removed_features = list()
        #features_copy = self.features.copy()
        from scipy.stats import pearsonr
        for feat in self.features:
            for featu in self.features[self.features.index(feat):]:
                pR, pn = pearsonr(self.X[feat],self.X[featu])
                if (pR > 0.9) and (feat != featu) and not (feat in featuresPR):
                    featuresPR.append(featu)
                    #print('remove ' + featu + ' corr ' + feat  )
        for feat in featuresPR:
            if feat in self.features:
                self.features.remove(feat)
                removed_features.append(feat)
        return removed_features
                    
    def FeatureSelection(self,train_size=0.5,model=RandomForestRegressor,param=None):
        self.train_size = train_size
        self.model = model
        max_ranking = 0.0
        min_ranking = 0.0
        self.selected_features = list()
        scores = list()
        #scores_a = np.zeros((len(self.features)))
        #n_features = {}
        X, y = make_regression(n_samples=len(result), n_features=len(self.features))
        for var in range(len(self.features)):
            X[:, var] = self.X[self.features[var]][:]
        y[:] = self.X[target][:]
        make_data_standard = False
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=train_size, shuffle=False)
        for i in range(5): #self.REPEAT_TIME):
            if param is None:
                model_est = self.model()
            else:
                model_est = self.model(**param)
            model_est = model()
            rfe = RFE(estimator=model_est,
                      n_features_to_select=1)
            rfe.fit(X_train, y_train)
            ranking = rfe.ranking_
            scores.append(ranking)
            max_ranking += max(ranking)
            min_ranking += 1        
            
        self.pToBeSelected = np.zeros((len(self.features)))
        for feature in self.anft:
            self.selected_features.append(self.features.index(feature))
            
        for i in range(len(self.features)):
            s = sum(scores[:])
            #s_max = max(s)
            self.pToBeSelected[i] = 1.0 - ((s[i]-min_ranking)/max_ranking)
        n_values = 0
        self.threshold = 0
        for feature in self.anft:
            value = self.pToBeSelected[self.features.index(feature)]
            if value == 1.0:
                print('TOO HIGH : ',feature)
            else:
                self.threshold += value
                n_values += 1
        self.threshold = self.threshold / n_values
        print('threhold : ', self.threshold)
        #self.threshold = np.mean( [ self.pToBeSelected[self.features.index('Load')],
        #                 self.pToBeSelected[self.features.index('Moisture in top oil')] ])
        #print( 'THRESHOLD :' + str(train_size) + ' ' + model.__name__ + ' ' + str(self.threshold))
        for i in range(len(self.features)):
            if self.pToBeSelected[i] >= self.threshold and (i not in self.selected_features):
                self.selected_features.append(i)  #TODO test > only.
        print(model.__name__ + '[' + str(train_size) + ']' + str(len(self.selected_features)))
        
    def TrainModel(self,param=None):
        if param is None:
            Tmodel = self.model()
        else:
            Tmodel = self.model(**param)
            self.param = param
        X, y = make_regression(n_samples=len(self.X), n_features=len(self.selected_features))
        for var in range(len(self.selected_features)):
            #print('add var : ', self.selected_features[var])
            X[:, var] = self.X[ self.features[ self.selected_features[var] ] ][:]
            #for x in range(len(self.X)):
            #    X[x, var] = self.X[ self.features[ self.selected_features[var] ] ][x]
        #for x in range(len(self.X)):
        y[:] = self.X[target][:]
        make_data_standard = True
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=self.train_size, shuffle=False)
        Tmodel.fit(X_train,y_train)
        y_pred = Tmodel.predict(X_test)
        self.estimator = Tmodel
        print('MAE test data [numbus of features:', X_train.shape[1],']: ' , mean_squared_error(y_test, y_pred))
        return Tmodel, y_pred
        
    def GridSearch(self,param_grid=None):
        if param_grid is None:
            return None
        self.grid_params = list()
        X, y = make_regression(n_samples=len(self.X), n_features=len(self.selected_features))
        for var in range(len(self.selected_features)):
            #for x in range(len(self.X)):
            X[:, var] = self.X[ self.features[ self.selected_features[var] ] ][:]
        #for x in range(len(self.X)):
        y[:] = self.X[target][:]
        make_data_standard = True
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=self.train_size, shuffle=False)
        for i in range(self.REPEAT_TIME):
            grid = GridSearchCV(self.model(),param_grid,#verbose=2,
                                cv=10,
                                refit=True,
                                scoring='neg_mean_squared_error',
                                n_jobs=4
                                )
            grid.fit(X_train,y_train)
            self.grid_params.append( grid.best_params_ )
        print(max(self.grid_params,
                  key=self.grid_params.count))
        return max(self.grid_params,key=self.grid_params.count)
                                                
    def printANFT(self):
        print(self.anft)
                    
    def printSF(self):
        print(self.selected_features)
        for feature in self.selected_features:
            print(self.features[feature], ' p:',
                  self.pToBeSelected[feature])
        
analytic_features = list()
analytic_features.append('Load')
analytic_features.append('Ambient temperature')
analytic_features.append('Moisture in top oil')
del result['DateTime']
del result['press']
del result['moist']
del result['100 metre U wind component']
del result['Unnamed: 0']
featSel = FeatureSelector(result,target='TOT',anal_feat=analytic_features)

featSel.CorrelationAnalysis()
removedFeatures = featSel.FeatureSelection(0.5)
featSel.printSF()
model, pred = featSel.TrainModel()
params = featSel.GridSearch(param_grid[RandomForestRegressor.__name__])
model, pred = featSel.TrainModel(param=params)
#featSel.printANFT()

sys.exit(0)
print('###################################################')
