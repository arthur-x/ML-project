# -*- coding: utf-8 -*-
#from datetime import datetime
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) 
'''
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
'''
def load_data_and_process(filename):
    #dataset = pd.read_csv(filename, parse_dates = [['year', 'month', 'day', 'hour']], header=0, index_col=0, date_parser=parse)
    dataset = pd.read_csv(filename)
    features = dataset[['month','day','hour','DEWP','HUMI','PRES','TEMP','cbwd','Iws','precipitation','Iprec','PM']]
    #dataset.index.name = 'date'
    values = features.values
    encoder = preprocessing.LabelEncoder()
    values[:, 7] = encoder.fit_transform(values[:, 7])
    # 所有feature 归一化到 0~1
    values = values.astype('float32')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    features[['month','day','hour','DEWP','HUMI','PRES','TEMP','cbwd','Iws','precipitation','Iprec','PM']] = scaled
    
    return features,scaler

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i)['PM'])
        if i == 0:
            names += [('var%d(t)' % (n_vars))]
        else:
            names += [('var%d(t+%d)' % (n_vars, i))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
       agg.dropna(inplace=True)
 
    #print(agg.head())
    training_samples, training_labels, testing_samples, testing_labels = training_testing_divide(agg, n_in, n_out, n_vars)
    return training_samples, training_labels, testing_samples, testing_labels

# 划分数据集，
def training_testing_divide(dataset, n_in, n_out, n_vars):

    values = dataset.values
    training_number = 24000
    training = values[:training_number, :]
    testing = values[training_number:, :]

    training_samples, training_labels = training[:, :-n_out], training[:, -n_out:]
    testing_samples, testing_labels = testing[:, :-n_out], testing[:, -n_out:]

    training_samples = training_samples.reshape((training_samples.shape[0], -1, n_vars))
    testing_samples = testing_samples.reshape((testing_samples.shape[0], -1, n_vars))
    return training_samples,training_labels,testing_samples,testing_labels

if __name__ == '__main__':
    features,scaler =load_data_and_process('ShanghaiPM.csv')
    history_length = 48
    predict_length = 1
    n_epochs = 10
    n_vars = features.shape[1]
    
    #Create supervised-learning data for LSTM
    training_samples, training_labels, testing_samples, testing_labels = series_to_supervised(features, history_length, predict_length, dropnan=True)
    #print(training_samples[1,:,:], training_labels[1,:], testing_samples[1,:,:], testing_labels[1,:])
    print(training_samples.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(n_vars, input_shape=(training_samples.shape[1], training_samples.shape[2]),return_sequences=True))
    model.add(LSTM(n_vars, return_sequences=False))
    model.add(Dense(predict_length))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(training_samples, training_labels, epochs=n_epochs, batch_size=24, verbose=2, shuffle=True)
    '''
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()
    '''
    test_batch = 100
    predict = model.predict(testing_samples[:test_batch,:,:])
    
    #predict = scaler.inverse_transform(predict)
    #ground = scaler.inverse_transform(testing_labels[:test_batch,:])
    plt.plot(testing_labels[:test_batch,:],label='actual pm')
    plt.plot(predict,label='predicted pm by LSTM')
    plt.legend()
    plt.show()
    '''
    training_predict = pd.DataFrame(model.predict(training_samples),columns=['predict'])
    testing_predict=pd.DataFrame(model.predict(testing_samples),columns=['predict'])
    predict=pd.concat([training_predict,testing_predict])
    predict.to_csv(r'./predict.csv', columns=['predict'], index=False, sep=',')

    testing_MAE=mean_absolute_error(testing_labels, testing_predict)
    print(testing_MAE)
    
    '''


