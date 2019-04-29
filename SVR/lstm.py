import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.preprocessing import StandardScaler
from get_data import gen_input
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


def get_train_and_test():
    x,y,scaler_speed, scaler_flow , scaler_occu = gen_input()

    x_train = x[:-10]
    x_test = x[-10:]
    y_train = y[:-10][:,:2]
    y_test = y[-10:][:,:2]

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)

    return x_train, x_test, y_train, y_test, scaler_speed, scaler_flow , scaler_occu

def train_lstm(x_train,y_train):
    model = Sequential()

    model.add(LSTM(units = 128, input_shape = (100,3), return_sequences = True))
    model.add(LSTM( 128, return_sequences=False)) # SET HERE
    # regressor.add(Dropout(0.2))
    model.add(Dense(2))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.summary()

    model.fit(x_train, y_train, epochs = 500, batch_size = 32)

    with open('lstm.pickle','wb') as f:
        pickle.dump(model,f)


def test_lstm(x_test,y_test,scaler_speed, scaler_flow,scaler_occu):
    with open('lstm.pickle','rb') as f:
        model = pickle.load(f)

    ## 只预测速度
    # pred=[]
    # for i in x_test:
    #     vec = np.array([i.tolist()])
    #     speed = model.predict(vec)
    #     pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][0])
    #
    # # print (pred)
    # x = list(range(len(y_test)))
    #
    # y_true = [scaler_speed.inverse_transform(i.reshape(1,1)).tolist()[0][0] for i in y_test]
    # # print (y_true)

    # 预测速度与流量
    pred=[]
    for i in x_test:
        # print('i',i,type(i))
        # vec = np.array([i.tolist()])
        # print (vec,type(vec))
        speed = model.predict(i.reshape(1,100,3))#vec)
        pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][:])

    pred_speed = [row[0] for row in pred]
    pred_flow = [row[1] for row in pred]

    true_speed = scaler_speed.inverse_transform(np.asarray(y_test)[:,0].reshape(-1,1)).transpose()[0]
    true_flow = scaler_flow.inverse_transform(np.asarray(y_test)[:,1].reshape(-1,1)).transpose()[0]

    print (pred_speed,'\n',true_speed,'\n\n',pred_flow,'\n',true_flow)

    x = list(range(len(pred_speed)))
    fig = plt.figure()

    plt.plot(x,pred_speed,linestyle = ':',color = 'r',label = 'Speed_predict')
    plt.plot(x,true_speed,linestyle = '-',color = 'g',label = 'Speed_true')

    plt.plot(x,pred_flow,linestyle = ':',color = 'b',label = 'flow_predict')
    plt.plot(x,true_flow,linestyle = '-',color = 'y',label = 'flow_true')
    plt.legend(loc='best',framealpha=0.5)
    plt.show()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, scaler_speed, scaler_flow , scaler_occu = get_train_and_test()
    train_lstm(x_train,y_train)
    test_lstm(x_test,y_test,scaler_speed, scaler_flow,scaler_occu)
