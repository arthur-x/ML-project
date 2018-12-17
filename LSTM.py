from datetime import datetime
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

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def load_data_and_process(filename):
    dataset = pd.read_csv(filename, parse_dates = [['year', 'month', 'day', 'hour']], header=0, index_col=0, date_parser=parse)
    
    dataset.index.name = 'date'
    '''
    #不考虑季节
    dataset=dataset.drop(columns='season')
    features=dataset.drop(columns=['PM'])
    # 把风向转换为数字
    values = features.values
    encoder = preprocessing.LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # 所有feature 归一化到 0~1
    values = values.astype('float32')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    dataset[['DEWP','HUMI','PRES','TEMP','cbwd','Iws','precipitation','Iprec']]=scaled
    #print(dataset.head())
    '''
    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled2 = scaler2.fit_transform(dataset['PM'].values.reshape(-1,1))
    plt.plot(scaled2[:24000])
    plt.show()
    dataset['PM'] = scaled2
    return dataset['PM']

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 #if type(data) is list else data.shape[1]
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
            names += ['var9(t)']
        else:
            names += [('var%d(t+%d)' % (9, i))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
       agg.dropna(inplace=True)
 
    #print(agg.head())
    training_samples, training_labels, testing_samples, testing_labels = training_testing_divide(agg, n_in, n_out)
    return training_samples, training_labels, testing_samples, testing_labels

# 划分数据集，
def training_testing_divide(dataset, n_in, n_out):

    values = dataset.values
    training_number = 24000
    training = values[:training_number, :]
    testing = values[training_number:, :]

    training_samples, training_labels = training[:, :-n_out], training[:, -n_out:]
    testing_samples, testing_labels = testing[:, :-n_out], testing[:, -n_out:]

    training_samples = training_samples.reshape((training_samples.shape[0], -1, 1))
    testing_samples = testing_samples.reshape((testing_samples.shape[0], -1, 1))
    return training_samples,training_labels,testing_samples,testing_labels

if __name__ == '__main__':
    dataset=load_data_and_process('ShanghaiPM.csv')
    history_length = 240
    predict_length = 120
    epochs = 10
    training_samples, training_labels, testing_samples, testing_labels = series_to_supervised(dataset, history_length, predict_length, dropnan=True)
    #print(training_samples[1,:,:], training_labels[1,:], testing_samples[1,:,:], testing_labels[1,:])
    print(training_samples.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(24, training_samples.shape[1], training_samples.shape[2]), stateful=True))
    model.add(Dense(predict_length))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    for i in range(epochs):
        model.fit(training_samples, training_labels, epochs=1, batch_size=24, verbose=0, shuffle=False)
        model.reset_states()
        print('epoch',i+1,'/',epochs)

    
    predict = model.predict(testing_samples[:24,:,:])
    plt.plot(testing_labels[10,:],label='actual pm')
    plt.plot(predict[10],label='predicted pm')
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


