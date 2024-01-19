#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# @author: iramirezg@mondraon.edu

import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor #base_estimator__ for AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE #Gives the order of elimination and handles recursivity
from sklearn.metrics import mean_squared_error #MSE

class FeatureSelector:
    def __init__(self,pddf,target=None,anal_feat=None):
        self.X = pddf
        self.param = None
        self.target = target
        self.features = list(self.X.keys())
        self.features.remove(self.target)
        self.anft = list()
        self.REPEAT_TIME = 5    #Times to repeat FeatureSelection or HyperparameterTuning process.
        if anal_feat is not None:
            for feat in anal_feat:
                if feat in self.X.keys():
                    self.anft.append(feat)
                    
    def CorrelationAnalysis(self,correlationThreshold=0.9):
        featuresPR = list()
        removed_features = list()
        from scipy.stats import pearsonr
        for feat in self.features:
            for featu in self.features[self.features.index(feat):]:
                pR, pn = pearsonr(self.X[feat],self.X[featu])
                if (pR > correlationThreshold) and (feat != featu) and not (feat in featuresPR) and not (featu in self.anft):
                    featuresPR.append(featu)
                    print('remove ' + featu + ' corr ' + feat + '  p:' + str(pR)  )
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
        X, y = make_regression(n_samples=len(self.X), n_features=len(self.features))
        for var in range(len(self.features)):
            X[:, var] = self.X[self.features[var]][:]
        y[:] = self.X[target][:]
        make_data_standard = False
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
        for i in range(self.REPEAT_TIME):
            if param is None:
                estimator = self.model()
            else:
                if self.model is AdaBoostRegressor:
                    estimator = self.model(base_estimator=DecisionTreeRegressor())
                    estimator.set_params(**param)
                else:
                    estimator = self.model(**param)      
            rfe = RFE(estimator=estimator,
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
        for i in range(len(self.features)):
            if self.pToBeSelected[i] >= self.threshold and (i not in self.selected_features):
                self.selected_features.append(i)
        print(model.__name__ + '[' + str(train_size) + ']' + str(len(self.selected_features)))
        
    def TrainModel(self,param=None):
        if param is None:
            Tmodel = self.model()
        else:
            if self.model is AdaBoostRegressor:
                estimator = self.model(base_estimator=DecisionTreeRegressor())
                estimator.set_params(**param)
            else:
                estimator = self.model(**param)
            Tmodel = estimator #(**param)
        X, y = make_regression(n_samples=len(self.X), n_features=len(self.selected_features))
        for var in range(len(self.selected_features)):
            X[:, var] = self.X[ self.features[ self.selected_features[var] ] ][:]
        y[:] = self.X[target][:]
        make_data_standard = False
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, shuffle=False)
        Tmodel.fit(X_train,y_train)
        y_pred = Tmodel.predict(X_test)
        self.estimator = Tmodel
        print('MAE test data [number of features:', X_train.shape[1],']: ' , mean_squared_error(y_test, y_pred))
        return Tmodel, y_pred
        
    def HyperparamTune(self,param_grid=None):
        if param_grid is None:
            return None
        grid_params = list()
        X, y = make_regression(n_samples=len(self.X), n_features=len(self.selected_features))
        for var in range(len(self.selected_features)):
            X[:, var] = self.X[ self.features[ self.selected_features[var] ] ][:]
        y[:] = self.X[target][:]
        make_data_standard = False
        if make_data_standard:
            scaler=StandardScaler()
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, shuffle=False)
        for i in range(self.REPEAT_TIME):
            if self.model is AdaBoostRegressor:
                estimator = self.model(base_estimator=DecisionTreeRegressor())
            else:
                estimator = self.model()
            grid = GridSearchCV(estimator,param_grid,cv=10, refit=True)
            grid.fit(X_train,y_train)
            grid_params.append( grid.best_params_ )
        return max(grid_params,key=grid_params.count)
                                                
    def printANFT(self):
        print(self.anft)
                    
    def printSF(self):
        print(self.selected_features)   #Print list of selected features, numerical order in CSV
        for feature in self.selected_features:
            print(self.features[feature], ' p:',    #Print feature name 
                  self.pToBeSelected[feature])      #with probability to be selected.
            
if __name__ == "__main__": 
    EstimatorModels = [ AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor ] #list of regressors to evaluate
    train_sizes = [0.5,0.75] #list of training sizes to evaluate
    param_grid = {} #Dictionary with regressor names as keywords and parameter grid as value
    param_grid[AdaBoostRegressor.__name__] = {
        'n_estimators' : [50,100,200,500],
        'base_estimator__min_samples_leaf' : [1,3,5,10,20],
        'base_estimator__max_leaf_nodes' : [None,3,5,10,20,50],
        'base_estimator__max_depth' : [None,3,10,20],
        }
    param_grid[RandomForestRegressor.__name__] = {
        'n_estimators' : [50,100,200,500],
        'min_samples_leaf' : [1,3,5,10,20],
        'max_leaf_nodes' : [None, 3,5, 10, 20,50],
        'max_depth' : [None,3,10,20],
        }
    param_grid[GradientBoostingRegressor.__name__] = {
        'learning_rate' : [0.5,0.7,0.9,1.0],    #With 1.0 (default) it is not random    
        'n_estimators' : [50,100,200,500],
        'min_samples_leaf' : [1,3,5,10,20],
        'max_leaf_nodes' : [None,3,5,10,20,50],
        'max_depth' : [None,3,10,20],
        }
    
    dataframe = pd.read_csv("tmp.csv",parse_dates=['DateTime']) #Load "pseudo-clean" data from CSV file.
    del dataframe['DateTime']  #Meaningless data variable for regression.
    dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()] #Remove constant columns.
    
    dataframe.dropna(axis='columns',how='any',inplace=True) #Remove columns with NaN values.
    
    target = 'TOT' #prediction target. Truth value.
    analytic_features = [ 'Load','Ambient temperature','Moisture in top oil' ] #MDTM analytic features
    #analytic_features = [ 'Load','Ambient temperature' ] #IEC analytic features
        
    featSel = FeatureSelector(dataframe,target='TOT',anal_feat=analytic_features) #Create FeatureSelector
    featSel.CorrelationAnalysis() #Do correlation analysis, removes correlated automatically
    for train_s in train_sizes:
        for estimator in EstimatorModels:
            print('TEST ', estimator.__name__, ' train size ', train_s)
            removedFeatures = featSel.FeatureSelection(train_s,model=estimator) #Feature selection
            featSel.printSF() #Print selected features with probability to be selected (result #1)
            model, pred = featSel.TrainModel() #Model training
            params = featSel.HyperparamTune(param_grid[estimator.__name__]) #Get best hyperparameters.
            print(params) #Print hyperparameters
            model, pred = featSel.TrainModel(param=params) #Model training
            removedFeatures = featSel.FeatureSelection(train_s,model=estimator,param=params) #Feature selection with hyperparams
            featSel.printSF() #Print selected features (may change from previous result #1)
