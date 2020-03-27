#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:25:18 2019

@author: akashteckchandani
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


file ='GOT_character_predictions.xlsx'

got_df = pd.read_excel(file)

######Regular exploration

got_df.info()
got_df.head()

######checking for null values
got_df.isnull().sum()

######dropping columns with significant amount of missing values
got = got_df.drop(['spouse',
              'mother', 
              'father',
              'name',
              'heir',
              'dateOfBirth',
              'isAliveMother',
              'isAliveFather',
              'isAliveHeir',
              'isAliveSpouse',
              'S.No',
              'culture',
              'isNoble',
              'isMarried'
              ], axis=1)

got.isnull().sum()

######Converting 'name' and 'house' to lowercase
got['house'] = got['house'].str.lower()

      
####### filling missing values of age column with mean. 
mean = got['age'].mean()
got['age'].fillna(mean, inplace = True)


## correlation matrix 
cor = got.corr()
cor["isAlive"].sort_values(ascending=False)

df_cor = got.corr().round(2)
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df_cor,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Akash_got_heatmap.png')
plt.show()

#######factorizing ordinal column title by giving random numbers in every value group
got.loc[:, "title"] = pd.factorize(got.title)[0]

####### filling missing values with -1. 
got.loc[:, "house"] = pd.factorize(got.house)[0]
got.columns = map(lambda x: x.replace(".", "").replace("_", ""), got.columns)
got.fillna(value = -1, inplace = True)
       
###### checking the amount of -1 of each column.   
got['title'].value_counts()
got['house'].value_counts()

got.isnull().sum()

""" plotting relationship between 'num dead relations' and people who are alive. 
Assuming that the people who have at least one relation have a higher tendency 
to live in the show. Thus we converted this into a binary column, 
if the correlation is grater than or equal to 1 then convert it into 1, else 0."""
    

for val in enumerate(got.loc[ : , 'numDeadRelations']):
     if val[1] >= 1 :
         got.loc[val[0],'numDeadRelations'] = 1
     else :
         got.loc[val[0],'numDeadRelations'] = 0

got['numDeadRelations'].value_counts()

######We are dropping the  column and setting the target and feature columns  


target = got.loc[:,'isAlive']
got = got.drop(['isAlive',],axis = 1)
features = got


###### train test split 
X_train, X_test, y_train, y_test = train_test_split(
                                    features,
                                    target,
                                    test_size = 0.1,
                                    random_state = 508,
                                    stratify = target)


###### K Neighbors classifier

parameters = {'n_neighbors' : range(1,18,2),
              'metric' : ['minkowski','manhattan','euclidean','chebyshev']}

grid = GridSearchCV(KNeighborsClassifier(),parameters)
model = grid.fit(features,target)
model

print(model.best_params_,model.best_estimator_,model.best_score_)



knn = KNeighborsClassifier(n_neighbors = 17,
                                          
                                          metric = 'manhattan')

####### Fitting the classifier to the training set

knn_fit = knn.fit(X_train, y_train)

#######Checking the accuracy of the model
knn_class_pred = knn_fit.predict(X_test)

knn_pred_prob = knn_fit.predict_proba(X_test)

#####AUC

knn_class_auc = roc_auc_score(y_test, knn_pred_prob[:, 1]).round(3)

print(knn_class_auc)

######### Cross viladation for KNN Classifier
crossv_knn = cross_val_score(knn,
                           features,
                           target,
                           cv = 3, scoring = 'roc_auc')


print(crossv_knn)


print('\nAverage: ',
      pd.np.mean(crossv_knn).round(3),
      '\nMinimum: ',
      min(crossv_knn).round(3),
      '\nMaximum: ',
      max(crossv_knn).round(3))



#######Random Forest
Rand_forest = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


####### Fitting the classifier to the training set
Rand_forest_fit = Rand_forest.fit(X_train, y_train)

#######Checking the accuracy of the model
print('Training Score', Rand_forest_fit.score(X_train, y_train).round(4))
print('Testing Score:', Rand_forest_fit.score(X_test, y_test).round(4))


######Assigning accuracy scores to variables
forest_fit_train = Rand_forest_fit.score(X_train, y_train)
forest_fit_test  = Rand_forest_fit.score(X_test, y_test)


forest_class_pred = Rand_forest_fit.predict(X_test)

forest_pred_prob = Rand_forest_fit.predict_proba(X_test)

###### AUC Score
auc_score = roc_auc_score(y_test, forest_pred_prob[:, 1]).round(3)

print(auc_score)

#######Cross Validation
crossv_gini = cross_val_score(Rand_forest,
                           features,
                           target,
                           cv = 3, scoring = 'roc_auc')

print(crossv_gini)
print('\nAverage: ',
      pd.np.mean(crossv_gini).round(3),
      '\nMinimum: ',
      min(crossv_gini).round(3),
      '\nMaximum: ',
      max(crossv_gini).round(3))




prediction_df = pd.DataFrame({'Actual' : y_test,
                                    'KNN Classifier' : knn_class_pred,
                                    'Random Forest' : forest_class_pred })
    
prediction_df.to_excel("survivors_pred.xlsx")


