# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt

# Load data
def load_csv_data(filename):
    file = pd.read_csv(filename)

    features = file[['month','day','hour','season','DEWP','HUMI','PRES',
                 'TEMP','Iws','precipitation','Iprec']]
    pm = file['PM']
    
    features = np.array(features)
    pm = np.array(pm)
    
    return features,pm


# RandomForest
def PredictRandomForest(features,pm):

    rf = RandomForestRegressor(n_estimators=10,criterion='mse',max_features=8,max_depth=20)
    scaler = StandardScaler().fit(features)
    standardized_features = scaler.transform(features) 
    
    X_train, X_test, y_train, y_test = train_test_split(standardized_features,pm,test_size=0.002,random_state = 0)
    
    '''
    params = {
              'max_depth': range(1,10),
              'max_features': range(1,3)} 
    gsearch = GridSearchCV(estimator=rf, 
                           param_grid=params,
                           cv = 5)
    gsearch.fit(standardized_features,pm)
    print(gsearch.best_score_)
    print(gsearch.best_estimator_)
    '''

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print('MSE:',mean_squared_error(y_test, y_pred))
    plt.plot(y_test,label='actual pm')
    plt.plot(y_pred,label='predicted pm')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    features,pm = load_csv_data('BeijingPM.csv')

    PredictRandomForest(features,pm)
