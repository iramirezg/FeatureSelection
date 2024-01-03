#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:53:47 2021

@author: iramirezg
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:35:44 2021

@author: iramirezg
"""

import MgepCode #Custom functions.
import pandas as pd
import numpy as np
import datetime
import os

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
np.seterr(all="ignore")

delta_T=60 #sampling rate, mins! -> 60 mins?

coord_latitude = 39.23417187122616
coord_longitude = -5.629151724502627
I_rate = 1587.7
# --------------------
# DATA READ
# --------------------
filename = "DataFiles/logfile_2021_05_28.csv" #FIRST FILE
miss_files = [] #['2021_08_01','2021_08_24']
era5_1 = "DataFiles/era5lake.nc"
era5_2 = "DataFiles/era5rain.nc"
era5_3 = "DataFiles/era5temp.nc"
era5_4 = "DataFiles/era5rad.nc"
era5_5 = "DataFiles/era5cloud.nc"
era5_6 = 'DataFiles/era5eva.nc'
SierraBravaFullDateTimes = MgepCode.extract_SierraBrava_DateTimes(filename,True,miss_files)
execution_date = datetime.datetime.now()
result_dir = 'results/ARWtr' + datetime.datetime.now().strftime('%Y%m%d%H%M')
#Create results dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_dir = result_dir + '/'
today_days = list()
for today_time in SierraBravaFullDateTimes:
    if not (today_time.date() in today_days):
        today_days.append(today_time.date())
today_days.sort()
tmpFDATA = 'RFE' + str(delta_T) + 'T.csv' #None #'DataFiles/ZZFastLoadCSVtest.tmp'
#os.remove(tmpFDATA)
print("Days : " + str(len(today_days)))
#print(today_days)
#longNames = ( MgepCode.extract_era5_variables(era5_1) )
#longNames.update( MgepCode.extract_era5_variables(era5_2) )
#longNames.update( MgepCode.extract_era5_variables(era5_3) )
if ( not (tmpFDATA is None) ) and os.path.exists(tmpFDATA):
    result = pd.read_csv(tmpFDATA)
    result['Datetime'] = pd.to_datetime(result['Datetime'])
else:
    FullData = MgepCode.extract_SierraBrava_Variables(filename,None,True,miss_files)
    del FullData['Date']
    del FullData['Time']
    del FullData['No.']
    FullData['Load'] = FullData['Current L1'] / I_rate
    del FullData['Current L1']
    del FullData['Current L2']
    del FullData['Current L3']
    del FullData['TW 1']
    del FullData['TW 2 ']
    del FullData['TW 3 ']
    result = pd.DataFrame.from_dict(FullData)
    result = result.rename(columns={'Oil temperature 2':'TOT',
                                    'Canal_1':'Tamb',
                                    'TW 4':'HST',
                                    'Canal_2':'H2ppmIsupose'})
    result['Datetime'] = pd.to_datetime(SierraBravaFullDateTimes)
    result.set_index('Datetime')
    #print(result.columns)
    newsamples = result.resample(str(delta_T) + 'T',on='Datetime').mean()
    newsamples.to_csv(tmpFDATA)
    result = pd.read_csv(tmpFDATA)
    os.remove(tmpFDATA)
    result['Datetime'] = pd.to_datetime(result['Datetime'])
    #print(result.head)
    era5data = MgepCode.extract_era5_datas(era5_1, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) )
    era5data.update( MgepCode.extract_era5_datas(era5_2, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) ))
    era5data.update( MgepCode.extract_era5_datas(era5_3, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) ))
    era5data.update( MgepCode.extract_era5_datas(era5_4, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) ))
    era5data.update( MgepCode.extract_era5_datas(era5_5, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) ))
    era5data.update( MgepCode.extract_era5_datas(era5_6, coord_longitude, coord_latitude, result['Datetime'] - datetime.timedelta(hours=1) ))
    for key in era5data.keys():
        result[key] = era5data[key]
    #print(result.head)
if (not (tmpFDATA is None)) and (not os.path.exists(tmpFDATA)):
    result.to_csv(tmpFDATA,index=False)

keys = list(result.keys())
for var in keys:
    if (var != 'Datetime') and (( max(result[var]) == min(result[var]) ) or np.isnan(sum(result[var]))):
        print('remove : ' + str(np.isnan(sum(result[var]))) + ' ' + var)
        del result[var]
target = 'TOT' 
#print(result.head)
features = list(result.keys())
#Adostutakoak kendu
features.remove('Datetime')
features.remove('HST')
features.remove('TOT')
features_iec = list(['Load','Tamb']) 
#features_ieee = list(['Load','Tamb','Oil Level','Total Active Power','ltlt','mn2t','lblt','sp','d2m']) #Very good for gradient boost (0.96)
#features_ieee = list(['Load','Tamb','Oil Level','Total Active Power','Total Power Factor PF','ltlt','mn2t','lblt','sp','d2m','alnip','e','tisr']) #Very good for
#features_custom = list(['Load','Tamb','Oil Level','Total Active Power','ltlt','lblt','mn2t']) #Very good for random forest (1.08)
#features_custom = list(['Tamb','Oil Level','Total Active Power','Total Power Factor PF','ltlt','mn2t','lblt','sp','d2m','alnip','e','tisr']) #Very good for
features_custom = list(['Load','Tamb','fdir']) #
#Arazoak Oil level-ekin, olioaren presio baita (presioa==temperatura?)

print(str(len(features))+':'+str(features))

###END LOAD

from sklearn.metrics import mean_absolute_error

# #params calculation of transient HST trafo & TOT
D_Theta_TO_R=69-15.74 #kelvin
D_Theta_HST_R=14.9 #kelvin
tau_TO=272.46 #min
tau_W=14.95 #min
R=11.64 # ratio load loss/no load loss
y=1.6 #cooling specific, standard
x=0.8 #cooling specific, standard
k11=0.73 #cooling specific, standard
k21=1.56 #cooling specific, standard
k22=1.65 #cooling specific, standard

vTOT=MgepCode.TOT_trans(result['Load'], result['Tamb'], D_Theta_TO_R, D_Theta_HST_R, tau_TO, tau_W, R, y, x, k11, k21, k22, delta_T)
TOT_mae = mean_absolute_error(vTOT,result[target])
# print(mean_absolute_error(vTOT,result[target]))

#fig,ax = plt.subplots(len(train_sizes),len(EstimatorModels))
#for ts in range(len(train_sizes)):
#    for mod in range(len(EstimatorModels)):
#        ax[ts,mod].plot(prediction_y_iec[train_sizes[ts]][EstimtorModels[mod]])

#print(prediction_y_iec)
import pylab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr, pearsonr


#BEGIN CORRELATION
corr_val = np.zeros(len(features))

for idx in range(len(corr_val)):
    print(features[idx])
    corr_val[idx],pvalue = pearsonr(result[features[idx]],result[target])

print(corr_val)
print(pearsonr(result['TOT'],result['Load']))
ind = np.argpartition(corr_val, -4)[-4:]
ind = ind[np.argsort(corr_val[ind])]
print(ind)
print(features[ind[0]])
print(features[ind[1]])
print(features[ind[2]])
print(features[ind[3]])

fig, ax = plt.subplots(2,2,sharex=True)
ax[0,0].scatter(result[target],result[features[ind[0]]])
ax[0,0].set_title(features[ind[0]])
ax[0,0].annotate("r = {:.2f}".format(corr_val[ind[0]]),xy=(0.1,0.9),xycoords=ax[0,0].transAxes)
ax[0,1].scatter(result[target],result[features[ind[1]]])
ax[0,1].set_title(features[ind[1]])
ax[0,1].annotate("r = {:.2f}".format(corr_val[ind[1]]),xy=(0.1,0.9),xycoords=ax[0,1].transAxes)
ax[1,0].scatter(result[target],result[features[ind[2]]])
ax[1,0].set_title(features[ind[2]])
ax[1,0].annotate("r = {:.2f}".format(corr_val[ind[2]]),xy=(0.1,0.9),xycoords=ax[1,0].transAxes)
ax[1,1].scatter(result[target],result[features[ind[3]]])
ax[1,1].set_title(features[ind[3]])
ax[1,1].annotate("r = {:.2f}".format(corr_val[ind[3]]),xy=(0.1,0.9),xycoords=ax[1,1].transAxes)
#plt.show()
plt.savefig(result_dir + 'correlation.eps')

fig,ax = plt.subplots(figsize=(12,6))
p0, = ax.plot(result['Datetime'],result[target],color='black')
ax.set_ylabel(target)
ax.yaxis.label.set_color(p0.get_color())
ax1 = ax.twinx()
p1, = ax1.plot(result['Datetime'],result[features[ind[0]]],color='b',alpha=.4)
ax1.set_ylabel(features[ind[0]])
ax1.yaxis.label.set_color(p1.get_color())
ax2 = ax.twinx()
p2, = ax2.plot(result['Datetime'],result[features[ind[1]]],color='g', alpha=.4)
ax2.set_ylabel(features[ind[1]])
ax2.yaxis.label.set_color(p2.get_color())
ax2.spines['right'].set_position(('outward', 50))
ax3 = ax.twinx()
p3, = ax3.plot(result['Datetime'],result[features[ind[2]]],color='y', alpha=.4)
ax3.set_ylabel(features[ind[2]])
ax3.yaxis.label.set_color(p3.get_color())
ax3.spines['right'].set_position(('outward',100))
ax4 = ax.twinx()
p4, = ax4.plot(result['Datetime'],result[features[ind[3]]],color='r', alpha=.4)
ax4.set_ylabel(features[ind[3]])
ax4.yaxis.label.set_color(p4.get_color())
ax4.spines['right'].set_position(('outward', 150))
fig.tight_layout()
#plt.show()
plt.savefig(result_dir + 'plot.eps')
#END CORRELATION

#BEGIN FEATURE SELECTION
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

train_sizes = [0.7,0.8]
EstimatorModels = [ GradientBoostingRegressor, LinearRegression, RandomForestRegressor ]
repeat_times = 1 #TODO check if correct
#prediction_days = [65]
prediction_folds = KFold(7) #7
make_data_standard=True
selected_features = {}
scores = {}
scores_a = {}
first_deleted = {}
importances = {}
n_features = {}
max_ranking = {}
min_ranking = {}
prediction_y_rfecv = {}
mae = {}
mae_rfe = {}
scaler=StandardScaler()

for train_size in train_sizes:
    scores[train_size] = {}
    max_ranking[train_size] = {}
    min_ranking[train_size] = {}
    selected_features[train_size] = {}
    mae[train_size] = {}
    mae_rfe[train_size] = {}
    for model in EstimatorModels:
        max_ranking[train_size][model] = 0.0
        min_ranking[train_size][model] = 0.0
        selected_features[train_size][model] = list()
        importances[model] = list()
        scores[train_size][model] = list()
        scores_a[model] = np.zeros((len(features)))
        #first_deleted[model] = np.zeros((len(features),len(prediction_days)))
        n_features[model] = {}
        X, y = make_regression(n_samples=len(result['Datetime']), n_features=len(features))
        #importances = None
        
        for var in range(len(features)):
            for x in range(len(result['Datetime'])):
                X[x, var] = result[features[var]][x]
        for x in range(len(result['Datetime'])):
            y[x] = result[target][x]
        if make_data_standard:
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=train_size, shuffle=False)
        if model in [LinearRegression, Lasso]:
            repeat_timer = 1
            mae[train_size][model] = np.zeros(1)
            mae_rfe[train_size][model] = np.zeros(1)
        else:
            repeat_timer = repeat_times
            mae[train_size][model] = np.zeros(repeat_times)
            mae_rfe[train_size][model] = np.zeros(repeat_times)
        for i in range(repeat_timer):
            if model in [RandomForestRegressor, LinearRegression]:
                model_est = model(n_jobs=-1)
            else:
                model_est = model()
            model_est.fit(X_train,y_train)
            y_pred = model_est.predict(X_test)
            mae[train_size][model][i] = mean_absolute_error(y_test, y_pred)
            if model in [RandomForestRegressor, LinearRegression]:
                model_est = model(n_jobs=-1)
            else:
                model_est = model()
            rfecv = RFECV(estimator=model_est, cv=prediction_folds,
                          scoring='neg_mean_squared_error', n_jobs=-1)
            rfecv.fit(X_train, y_train)
            y_pred = rfecv.predict(X_test)
            mae_rfe[train_size][model][i] = mean_absolute_error(y_test, y_pred)
            
            #score = rfecv.score(, y)
            ranking = rfecv.ranking_
            #TODOn_features[model].append(rfecv.n_features_)
            #print('Number of features: ' + str(rfecv.n_features_) +
            #      ' with score of:' + str(score))
            # for i in range(len(features)):
            #     if rfecv.ranking_[i] == 1:
            #         scores_a[model][i] += 1
            #     if rfecv.ranking_[i] == np.max(rfecv.ranking_):
            #         first_deleted[model][i] += 1
            importances[model].append(rfecv.grid_scores_)
            #scores[model].append( score )
            scores[train_size][model].append(ranking)
            print(ranking)
            max_ranking[train_size][model] += max(ranking)
            min_ranking[train_size][model] += 1
        if model in [LinearRegression, Lasso]:
            print(scores_a[model][:])
            #print(mae[train_size][model.__name__][:])
        #else:
        #    scores_a[model][:] /= (repeat_times)
            #mae[train_size][model.__name__][:] /= (repeat_times)
#END FEATURE SELECTION
# for model in EstimatorModels:
#     selected_features[model] = list() 
#     for f in range(len(features)):
#         if scores_a[model][f] >= 0.8:
#             print(model.__name__ + ' ' + features[f] + '[' + str(f) + ']:' + str(scores_a[model][f]))
#             selected_features[model].append(f)
# print(selected_features)
#END FEATURE SELECTION
pToBeSelected = {}
selected_mix_features = list()
for train_size in train_sizes:
    pToBeSelected[train_size] = {}
    for model in EstimatorModels:
        pToBeSelected[train_size][model.__name__] = np.zeros((len(features)))
        for i in range(len(features)):
            s = sum(scores[train_size][model][:])
            s_max = max(s)
            pToBeSelected[train_size][model.__name__][i] = 1.0 - ((s[i]-min_ranking[train_size][model])/max_ranking[train_size][model])
            if pToBeSelected[train_size][model.__name__][i] > 0.8:
                selected_features[train_size][model].append(i)
            if (pToBeSelected[train_size][model.__name__][i] > 0.85) and (i not in selected_mix_features):
                selected_mix_features.append(i)
        print(model.__name__ + '[' + str(train_size) + ']' + str(len(selected_features[train_size][model])))

print(mae)
print(mae_rfe)

mae_sel = {}
for train_size in train_sizes:
    mae_sel[train_size] = {}
    for model in EstimatorModels:
        importances[model] = list()
        scores[train_size][model] = list()
        #scores_a[model] = np.zeros((len(features),len(prediction_days)))
        n_features[model] = {}
        X, y = make_regression(n_samples=len(result['Datetime']), n_features=len(selected_features[train_size][model]))
        #importances = None
        
        for var in range(len(selected_features[train_size][model])):
            for x in range(len(result['Datetime'])):
                X[x, var] = result[features[selected_features[train_size][model][var]]][x]
        for x in range(len(result['Datetime'])):
            y[x] = result[target][x]
        if make_data_standard:
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=train_size, shuffle=False)
        if model in [LinearRegression, Lasso]:
            repeat_timer = 1
            mae_sel[train_size][model] = np.zeros(1)
        else:
            repeat_timer = repeat_times
            mae_sel[train_size][model] = np.zeros(repeat_times)
        for i in range(repeat_timer):
            if model in [RandomForestRegressor, LinearRegression]:
                model_est = model(n_jobs=-1)
            else:
                model_est = model()
            #rfecv = RFECV(estimator=model_est, cv=prediction_folds,
            #              scoring='neg_mean_squared_error', n_jobs=-1)
            model_est.fit(X_train, y_train)
            y_pred = model_est.predict(X_test)
            mae_sel[train_size][model][i] = mean_absolute_error(y_test, y_pred)

mae_sel2 = {}
for train_size in train_sizes:
    mae_sel2[train_size] = {}
    for model in EstimatorModels:
        importances[model] = list()
        scores[train_size][model] = list()
        #scores_a[model] = np.zeros((len(features),len(prediction_days)))
        n_features[model] = {}
        X, y = make_regression(n_samples=len(result['Datetime']), n_features=len(selected_mix_features))
        #importances = None
        
        for var in range(len(selected_mix_features)):
            for x in range(len(result['Datetime'])):
                X[x, var] = result[features[selected_mix_features[var]]][x]
        for x in range(len(result['Datetime'])):
            y[x] = result[target][x]
        if make_data_standard:
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=train_size, shuffle=False)
        if model in [LinearRegression, Lasso]:
            repeat_timer = 1
            mae_sel2[train_size][model] = np.zeros(1)
        else:
            repeat_timer = repeat_times
            mae_sel2[train_size][model] = np.zeros(repeat_times)
        for i in range(repeat_timer):
            if model in [RandomForestRegressor, LinearRegression]:
                model_est = model(n_jobs=-1)
            else:
                model_est = model()
            #rfecv = RFECV(estimator=model_est, cv=prediction_folds,
            #              scoring='neg_mean_squared_error', n_jobs=-1)
            model_est.fit(X_train, y_train)
            y_pred = model_est.predict(X_test)
            mae_sel2[train_size][model][i] = mean_absolute_error(y_test, y_pred)

mae_sel0 = {}
selected_iec_features = ('Load','Tamb')
for train_size in train_sizes:
    mae_sel0[train_size] = {}
    for model in EstimatorModels:
        importances[model] = list()
        scores[train_size][model] = list()
        #scores_a[model] = np.zeros((len(features),len(prediction_days)))
        n_features[model] = {}
        #X, y = make_regression(n_samples=len(result['Datetime']), n_features=len(selected_iec_features))
        X = np.zeros((len(result['Datetime']), len(selected_iec_features)))
        y = np.zeros(len(result['Datetime']))
        #importances = None
        
        for var in range(len(selected_iec_features)):
            for x in range(len(result['Datetime'])):
                X[x, var] = result[selected_iec_features[var]][x]
        for x in range(len(result['Datetime'])):
            y[x] = result[target][x]
        if make_data_standard:
            SX = scaler.fit_transform(X, y)
        else:
            SX = X
        X_train, X_test, y_train, y_test = train_test_split(SX, y, train_size=train_size, shuffle=False)
        if model in [LinearRegression, Lasso]:
            repeat_timer = 1
            mae_sel0[train_size][model] = np.zeros(1)
        else:
            repeat_timer = repeat_times
            mae_sel0[train_size][model] = np.zeros(repeat_times)
        for i in range(repeat_timer):
            if model in [RandomForestRegressor, LinearRegression]:
                model_est = model(n_jobs=-1)
            else:
                model_est = model()
            #rfecv = RFECV(estimator=model_est, cv=prediction_folds,
            #              scoring='neg_mean_squared_error', n_jobs=-1)
            model_est.fit(X_train, y_train)
            y_pred = model_est.predict(X_test)
            mae_sel0[train_size][model][i] = mean_absolute_error(y_test, y_pred)
            
print(mae_sel)

for train_size in train_sizes:
    print(train_size)
    for model in EstimatorModels:
        print(model.__name__ )
        print(np.min(mae[train_size][model]),end='\t\t')
        print(np.max(mae[train_size][model]),end='\t\t')
        print(np.mean(mae[train_size][model]))
        print(np.min(mae_rfe[train_size][model]),end='\t\t')
        print(np.max(mae_rfe[train_size][model]),end='\t\t')
        print(np.mean(mae_rfe[train_size][model]))
        print(np.min(mae_sel[train_size][model]),end='\t\t')
        print(np.max(mae_sel[train_size][model]),end='\t\t')
        print(np.mean(mae_sel[train_size][model]))
        print(np.min(mae_sel2[train_size][model]),end='\t\t')
        print(np.max(mae_sel2[train_size][model]),end='\t\t')
        print(np.mean(mae_sel2[train_size][model]))
        print(np.min(mae_sel0[train_size][model]),end='\t\t')
        print(np.max(mae_sel0[train_size][model]),end='\t\t')
        print(np.mean(mae_sel0[train_size][model]))

print(len(selected_mix_features))

from scipy.stats import pearsonr

LoadColumn = np.zeros(len(selected_mix_features))
for f in range(len(selected_mix_features)):
    LoadColumn[f],pp = pearsonr(result['Load'],result[features[selected_mix_features[f]]])
#LoadCorrDF = pd.DataFrame(LoadCorr)
LoadCorr = pd.DataFrame(selected_mix_features)
LoadCorr['r'] = LoadColumn

LoadCorr = LoadCorr.sort_values('r',ascending=False)

import seaborn as sb

fig,ax = plt.subplots(figsize=(12,10))
varFC = {}
#for var in selected_mix_features:
for var in LoadCorr[0]:
    varFC[var] = result[features[var]] #[x]result[var]
df = pd.DataFrame(varFC)
corr = df.corr()
#corr = df.corr().sort_values(5,ascending=False)
#corr_d = corr.sort_values(by=['5'],ascending=False)
#sb.heatmap(corr.sort_values(by=5,ascending=False), cmap='coolwarm')
sb.heatmap(corr, cmap='coolwarm')
pylab.show()
plt.savefig(result_dir + 'heatmap.svg',bbox_inches='tight',dpi=600)


#print(pearsonr(result[target],result['Load']))
# #TOT, iec TOT and prediction
# fig,ax = plt.subplots(len(EstimatorModels),len(train_sizes),sharex=True,sharey=True,figsize=(12,9))
# for ts in train_sizes:
#     print('TRAIN size ' + str(ts) + ' :' + str(len(prediction_y_iec[ts][EstimatorModels[0].__name__][0]) ) )
#     ax[0,train_sizes.index(ts)].set_title(ts)
#     # fig = plt.figure()
#     # ax = fig.add_axes([0,0,1,1])
#     for model in range(len(EstimatorModels)):
#         y_len = len(prediction_y_iec[train_size][EstimatorModels[model].__name__][0])
#         for i in range(len(prediction_y_iec[train_size][EstimatorModels[model].__name__])):
#             ax[model,train_sizes.index(ts)].plot(result['Datetime'][-y_len:],prediction_y_iec[train_size][EstimatorModels[model].__name__][i],color='b',alpha=0.2)
#         ax[model,train_sizes.index(ts)].plot(result['Datetime'][-y_len:],result[target][len(result[target])-y_len:],color='g',label='TOT')
#         ax[model,train_sizes.index(ts)].plot(result['Datetime'][-y_len:],vTOT[len(vTOT)-y_len:],color='r',label='IEC')
#         ax_mae = df_mae['RFECV_'+str(ts)][EstimatorModels[model].__name__]
#         ax_mae1 = mae[ts][EstimatorModels[model].__name__][4,0]
#         print(df_mae['RFECV_'+str(ts)][EstimatorModels[model].__name__])
#         ax[model,train_sizes.index(ts)].annotate("MAE = {:.3f}".format(float(ax_mae1)),xy=(0.1,0.9),xycoords=ax[model,train_sizes.index(ts)].transAxes)
#     #ax.ticks()
# #fig.set_ylabel('Temp [deg C.]')
#     #ax.set_ylabel('Temp [deg C.]')
#     #locator = plt.MaxNLocator(10)
#     #ax.xaxis.set_major_locator(locator)
#     #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# pylab.draw()
# ax[2,0].set_xticklabels(ax[2,0].get_xticklabels(),rotation=45)
# #ax[2,0].annotate("MAE = {:.3f}".format(df_mae['IECF_'+ts][EstimatorModels[model].__name__]),xy=(0.1,0.9),xycoords=ax[2,0].transAxes)
# ax[2,1].set_xticklabels(ax[2,1].get_xticklabels(),rotation=45)
# fig.tight_layout()
#     #fig.autofmg_xdate()
# plt.savefig(result_dir + 'prediec.png',bbox_inches='tight')
