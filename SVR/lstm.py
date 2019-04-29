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


def get_train_and_test(type_list):
    if len(type_list) > 2:
        print ('too much parameters were given')

    x,y,scaler_speed, scaler_flow , scaler_occu = gen_input()
    x_train = x[:-10]
    x_test = x[-10:]

    if len(type_list) == 2:
        y_train = y[:-10][:,:2]
        y_test = y[-10:][:,:2]

    if len(type_list)==1:
        if type_list[0] == 'speed':
            y_train = y[:-10][:,0]
            y_test = y[-10:][:,0]
        if type_list[0] == 'flow':
            y_train = y[:-10][:,1]
            y_test = y[-10:][:,1]

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)

    return x_train, x_test, y_train, y_test, scaler_speed, scaler_flow , scaler_occu

def train_lstm(x_train,y_train,type_list):
    model = Sequential()
    model.add(LSTM(units = 128, input_shape = (100,3), return_sequences = True))
    model.add(LSTM( 128, return_sequences=False)) # SET HERE
    model.add(Dropout(0.2))

    if len(type_list) > 2:
        print ('too much parameters were given to train LSTM')
    elif len(type_list)==2:
        pickle_name = type_list[0]+'_'+type_list[1]
        model.add(Dense(2))
    elif len(type_list)==1:
        pickle_name = type_list[0]
        model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.summary()

    model.fit(x_train, y_train, epochs = 500, batch_size = 32)

    with open('lstm_'+ pickle_name +'.pickle','wb') as f:
        pickle.dump(model,f)


def test_lstm(x_test,y_test,scaler_speed, scaler_flow,scaler_occu,type_list):

    if len(type_list) > 2:
        print ('too much parameters were given to train LSTM')
    elif len(type_list)==2:
        pickle_name = type_list[0]+'_'+type_list[1]

        with open('lstm_' + pickle_name + '.pickle','rb') as f:
            model = pickle.load(f)

        pred=[]
        for i in x_test:
            # print('i',i,type(i))
            # vec = np.array([i.tolist()])
            # print (vec,type(vec))
            speed = model.predict(i.reshape(1,100,3))
            pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][:])

        x = list(range(len(pred)))

        pred_speed = [row[0] for row in pred]
        pred_flow = [row[1] for row in pred]

        true_speed = scaler_speed.inverse_transform(np.asarray(y_test)[:,0].reshape(-1,1)).transpose()[0]
        true_flow = scaler_flow.inverse_transform(np.asarray(y_test)[:,1].reshape(-1,1)).transpose()[0]

        print (pred_speed,'\n',true_speed,'\n\n',pred_flow,'\n',true_flow)

        fig = plt.figure()
        plt.plot(x,pred_speed,linestyle = ':',color = 'r',label = 'Speed_predict')
        plt.plot(x,true_speed,linestyle = '-',color = 'g',label = 'Speed_true')

        plt.plot(x,pred_flow,linestyle = ':',color = 'b',label = 'flow_predict')
        plt.plot(x,true_flow,linestyle = '-',color = 'y',label = 'flow_true')
        plt.legend(loc='best',framealpha=0.5)
        plt.title('LSTM prediction for speed and flow')
        plt.show()


    elif len(type_list)==1:
        pickle_name = type_list[0]

        with open('lstm_' + pickle_name + '.pickle','rb') as f:
            model = pickle.load(f)

        if type_list[0] == 'speed':
            pred=[]
            for i in x_test:
                speed = model.predict(i.reshape(1,100,3))
                pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][0])

            # print (pred)
            x = list(range(len(y_test)))
            y_true = [scaler_speed.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

            fig = plt.figure()
            plt.plot(x,pred,linestyle = ':',color = 'r',label = 'Speed_predict')
            plt.plot(x,y_true,linestyle = '-',color = 'g',label = 'Speed_true')
            plt.legend(loc='best',framealpha=0.5)
            plt.title('LSTM prediction for speed')
            plt.ylabel('speed')
            plt.show()

        if type_list[0] == 'flow':
            pred=[]
            for i in x_test:
                flow = model.predict(i.reshape(1,100,3))
                pred.append(scaler_flow.inverse_transform(flow.tolist()).tolist()[0][0])

            # print (pred)
            x = list(range(len(y_test)))
            y_true = [scaler_flow.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

            fig = plt.figure()
            plt.plot(x,pred,linestyle = ':',color = 'r',label = 'flow_predict')
            plt.plot(x,y_true,linestyle = '-',color = 'g',label = 'flow_true')
            plt.legend(loc='best',framealpha=0.5)
            plt.title('LSTM prediction for flow')
            plt.ylabel('flow')
            plt.show()





if __name__ == '__main__':
    type_list = ['speed','flow']
    x_train, x_test, y_train, y_test, scaler_speed, scaler_flow , scaler_occu = get_train_and_test(type_list)
    # train_lstm(x_train,y_train,type_list)
    test_lstm(x_test,y_test,scaler_speed, scaler_flow,scaler_occu,type_list)
