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

pd.set_option('display.max_columns', None) #不限制print的列数
#pd.set_option('display.max_rows', None)

# 这个函数在load_data(filename)函数里面有用
# 中间函数不用管
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def load_data_and_process(filename):
    dataset = pd.read_csv(filename, parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.index.name = 'date'
    #不考虑季节
    dataset=dataset.drop(columns='season')
    features=dataset.drop(columns=['PM'])
    # 把风向转换为数字
    values = features.values
    encoder = preprocessing.LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # 所有feature 归一化到0-1
    values = values.astype('float32')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    dataset[['DEWP','HUMI','PRES','TEMP','cbwd','Iws','precipitation','Iprec']]=scaled
    #将PM_1 表示PM(t+1)
    dataset.at[:,"PM_1"]=dataset['PM']
    dataset['PM_1']=dataset['PM_1'].shift(-1)
    dataset=dataset.drop([dataset.last_valid_index()])


    return dataset

# 划分数据集，
def training_testing_divide(dataset):

    values = dataset.values
    training_number = int(0.5* (values.shape[0]))
    training = values[:training_number, :]
    testing = values[training_number:, :]

    training_samples, training_labels = training[:, :-1], training[:, -1]
    testing_samples, testing_labels = testing[:, :-1], testing[:, -1]

    training_samples = training_samples.reshape((training_samples.shape[0], 1, training_samples.shape[1]))
    testing_samples = testing_samples.reshape((testing_samples.shape[0], 1, testing_samples.shape[1]))
    return training_samples,training_labels,testing_samples,testing_labels

if __name__ == '__main__':
    dataset=load_data_and_process('GuangzhouPM20100101_20151231_withoutNA.csv')
    #plot_features(dataset)
    training_samples, training_labels, testing_samples, testing_labels=training_testing_divide(dataset)

    # design network
    model = Sequential()
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(training_samples, training_labels, epochs=25, batch_size=18, verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    training_predict = pd.DataFrame(model.predict(training_samples),columns=['predict'])
    testing_predict=pd.DataFrame(model.predict(testing_samples),columns=['predict'])
    predict=pd.concat([training_predict,testing_predict])
    predict.to_csv(r'./predict.csv', columns=['predict'], index=False, sep=',')

    testing_MAE=mean_absolute_error(testing_labels, testing_predict)
    print(testing_MAE)



