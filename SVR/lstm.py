import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.preprocessing import StandardScaler
from utils import get_LSTM_input
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def get_train_and_test(up_fname,down_fname,type_list):
    if len(type_list) > 3:
        print ('too much parameters were given')
        return None

    x, y, scalers = get_LSTM_input(up_fname,down_fname)
    last_num = 20
    x_train = x[:-last_num]
    x_test = x[-last_num:]

    if len(type_list)>1:
        y_train = y[:-last_num][:,:len(type_list)]
        y_test = y[-last_num:][:,:len(type_list)]

    if len(type_list)==1:
        if type_list[0] == 'speed':
            y_train = y[:-last_num][:,0]
            y_test = y[-last_num:][:,0]
        if type_list[0] == 'flow':
            y_train = y[:-last_num][:,1]
            y_test = y[-last_num:][:,1]
        if type_list[0] == 'occu':
            y_train = y[:-last_num][:,2]
            y_test = y[-last_num:][:,2]

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)

    return x_train, x_test, y_train, y_test, scalers

def train_lstm(x_train,y_train,type_list):
    model = Sequential()
    # model.add(LSTM(units = 64, input_shape = (100,3), return_sequences = True))
    model.add(LSTM(units=32, input_shape = (100,2), return_sequences = True))
    if (type_list[0]=='speed'):
        model.add(LSTM(units=32, input_shape = (100,2), return_sequences = True))
    model.add(LSTM(32, return_sequences=False)) # SET HERE

    if len(type_list) > 3:
        print ('too much parameters were given to train LSTM')
    elif len(type_list)==3:
        pickle_name = type_list[0]+'_'+type_list[1]+'_'+type_list[2]
        model.add(Dense(3))
    elif len(type_list)==2:
        pickle_name = type_list[0]+'_'+type_list[1]
        model.add(Dense(2))
    elif len(type_list)==1:
        pickle_name = type_list[0]
        model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.summary()

    model.fit(x_train, y_train, epochs = 100, batch_size = 32)

    with open('lstm_'+ pickle_name +'.pickle','wb') as f:
        pickle.dump(model,f)


def test_lstm(x_test,y_test,scaler_speed,scaler_flow,scaler_occu,type_list):

    if len(type_list) > 3:
        print ('too much parameters were given to train LSTM')

    elif len(type_list)==3:
        pickle_name = type_list[0]+'_'+type_list[1]+'_'+type_list[2]


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
        pred_occu = [row[2] for row in pred]

        true_speed = scaler_speed.inverse_transform(np.asarray(y_test)[:,0].reshape(-1,1)).transpose()[0]
        true_flow = scaler_flow.inverse_transform(np.asarray(y_test)[:,1].reshape(-1,1)).transpose()[0]
        true_occu = scaler_occu.inverse_transform(np.asarray(y_test)[:,2].reshape(-1,1)).transpose()[0]


        fig = plt.figure()
        plt.plot(x,pred_speed,linestyle = ':',color = 'r',label = 'Speed_predict')
        plt.plot(x,true_speed,linestyle = '-',color = 'r',label = 'Speed_true')

        plt.plot(x,pred_flow,linestyle = ':',color = 'g',label = 'Flow_predict')
        plt.plot(x,true_flow,linestyle = '-',color = 'g',label = 'Flow_true')

        plt.plot(x,pred_occu,linestyle = ':',color = 'b',label = 'Occu_predict')
        plt.plot(x,true_occu,linestyle = '-',color = 'b',label = 'Occu_true')

        plt.legend(loc='best',framealpha=0.5)
        plt.title('LSTM prediction for speed,flow and occu')
        plt.show()

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
        plt.plot(x,true_speed,linestyle = '-',color = 'r',label = 'Speed_true')

        plt.plot(x,pred_flow,linestyle = ':',color = 'g',label = 'flow_predict')
        plt.plot(x,true_flow,linestyle = '-',color = 'g',label = 'flow_true')
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
                speed = model.predict(i.reshape(1,100,2))
                pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][0])

            # print (pred)
            x = list(range(len(y_test)))
            y_true = [scaler_speed.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

            print ('mae: ',mean_absolute_error(y_true,pred))
            print ('r2 score: ',r2_score(y_true,pred))
            print ('mse: ',mean_squared_error(y_true,pred))
            print ('mape: ',MAPE(y_true,pred))

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
                # flow = model.predict(i.reshape(1,100,3))
                flow = model.predict(i.reshape(1,100,2))
                pred.append(scaler_flow.inverse_transform(flow.tolist()).tolist()[0][0])

            # print (pred)
            x = list(range(len(y_test)))
            y_true = [scaler_flow.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

            print ('mae: ',mean_absolute_error(y_true,pred))
            print ('r2 score: ',r2_score(y_true,pred))
            print ('mse: ',mean_squared_error(y_true,pred))
            print ('mape: ',MAPE(y_true,pred))

            fig = plt.figure()
            plt.plot(x,pred,linestyle = ':',color = 'r',label = 'flow_predict')
            plt.plot(x,y_true,linestyle = '-',color = 'g',label = 'flow_true')
            plt.legend(loc='best',framealpha=0.5)
            plt.title('LSTM prediction for flow')
            plt.ylabel('flow')
            plt.show()

        # if type_list[0] == 'occu':
        #     pred=[]
        #     for i in x_test:
        #         flow = model.predict(i.reshape(1,100,3))
        #         pred.append(scaler_flow.inverse_transform(flow.tolist()).tolist()[0][0])
        #
        #     # print (pred)
        #     x = list(range(len(y_test)))
        #     y_true = [scaler_flow.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]
        #
        #     print ('mae: ',mean_absolute_error(y_true,pred))
        #     print ('r2 score: ',r2_score(y_true,pred))
        #     print ('mse: ',mean_squared_error(y_true,pred))
        #     print ('mape: ',MAPE(y_true,pred))
        #
        #     fig = plt.figure()
        #     plt.plot(x,pred,linestyle = ':',color = 'r',label = 'occu_predict')
        #     plt.plot(x,y_true,linestyle = '-',color = 'g',label = 'occu_true')
        #     plt.legend(loc='best',framealpha=0.5)
        #     plt.title('LSTM prediction for occu')
        #     plt.ylabel('occu')
        #     plt.show()





if __name__ == '__main__':
    type_list = ['speed']
    x_train, x_test, y_train, y_test, scalers = get_train_and_test('train_up','train_down',type_list)
    train_lstm(x_train,y_train,type_list)
    scaler_speed, scaler_flow, scaler_occu = scalers['down_speed'],scalers['down_flow'],scalers['down_occu']
    test_lstm(x_test,y_test,scaler_speed, scaler_flow,scaler_occu,type_list)
