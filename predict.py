# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt

# Load data
def load_csv_data(filename):
    file = pd.read_csv(filename)

    data = file[['year','month','day','hour','season','PM_Jingan',
                 'PM_US Post','PM_Xuhui','DEWP','HUMI','PRES',
                 'TEMP','cbwd','Iws','precipitation','Iprec']]
    data = data.dropna(subset = set(data.columns) - {'PM_Jingan',
                       'PM_US Post','PM_Xuhui'})
    data = data.dropna(thresh = 1, 
                       subset = {'PM_Jingan','PM_US Post','PM_Xuhui'})
    print(len(data))
    data = data.dropna(subset = {'PM_US Post'})
    print(len(data))
    data = np.array(data)
    return data


# RandomForest
def testRandomForest(features, labels, features_test):

    rf = RandomForestClassifier(n_estimators = 200, criterion='entropy', max_depth=44, max_features=28)
    scaler = StandardScaler().fit(features)
    standardized_features = scaler.transform(features) 
    
    X_train, X_test, y_train, y_test = train_test_split(standardized_features,labels,test_size=0.2,random_state = 0)
    '''
    params = {'criterion': ['gini','entropy'],
              'max_depth': range(10,50),
              'max_features': range(10,50)} 
    gsearch = GridSearchCV(estimator=rf, 
                           param_grid=params,
                           cv = 5)
    gsearch.fit(standardized_features,labels)
    print(gsearch.best_score_)
    print(gsearch.best_estimator_)
    '''

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print('accuracy:',accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    data = load_csv_data('ShanghaiPM20100101_20151231.csv')

    #testRandomForest(features, labels, features_test)
