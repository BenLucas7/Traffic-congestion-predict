import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import MinMaxScaler


def read_excel(fname):
    df = pd.read_excel('data/'+fname+'.xlsx',index_col = 0)
    df.index = pd.to_datetime(df.index)
    timeseries = df.index.values

    scaler_speed = MinMaxScaler()
    scaler_flow = MinMaxScaler()
    scaler_occu = MinMaxScaler()

    # print (np.array(df['speed']).reshape(-1,1))
    speed = scaler_speed.fit_transform(np.array(df['speed']).reshape(-1,1))
    flow = scaler_flow.fit_transform(np.array(df['flow']).reshape(-1,1))
    occu = scaler_occu.fit_transform(np.array(df['occu']).reshape(-1,1))

    return_dict = {}
    return_dict['speed'] = speed
    return_dict['flow'] = flow
    return_dict['occu'] = occu
    return_dict['scaler_speed'] = scaler_speed
    return_dict['scaler_flow'] = scaler_flow
    return_dict['scaler_occu'] = scaler_occu
    return_dict['time'] = timeseries

    return return_dict

def get_SVR_input(fname,type='speed'):
    # 没有归一化效果更好

    excel_vec = read_excel(fname)

    speed_list = excel_vec['scaler_speed'].inverse_transform(excel_vec['speed']).transpose().tolist()[0]
    flow_list = excel_vec['scaler_flow'].inverse_transform(excel_vec['flow']).transpose().tolist()[0]

    # speed_list = excel_vec['speed'].transpose().tolist()[0]
    # flow_list = excel_vec['flow'].transpose().tolist()[0]
    date = excel_vec['time']

    x = [] # [speed123,flow123,occu123]
    y_speed = []
    y_flow = []
    time_log = []

    for i in range(0,len(speed_list),4):
        try:
            vec = [speed_list[i],speed_list[i+1],speed_list[i+2]]
            vec.extend([flow_list[i],flow_list[i+1],flow_list[i+2]])
            x.append(vec)
            y_speed.append(speed_list[i+3])
            y_flow.append(flow_list[i+3])
            time_log.append(date[i+3]) # just for plot

        except IndexError:
            break

    # print ('x:\n',x,'\n','speed:\n',y_speed,'\n','flow:\n',y_flow,'\n','\ntime:\n',time_log)
    if type=='speed':
        return np.asarray(x), np.asarray(y_speed),np.asarray(time_log),excel_vec['scaler_speed']
    if type=='flow':
        return np.asarray(x), np.asarray(y_flow),np.asarray(time_log),excel_vec['scaler_flow']


    return EOFError

def get_LSTM_input(up_fname,down_fname,test_mode=False):

    vec_up = read_excel(up_fname)
    speed_up = vec_up['speed'].transpose().tolist()[0]
    flow_up = vec_up['flow'].transpose().tolist()[0]
    occu_up = vec_up['occu'].transpose().tolist()[0]
    date = vec_up['time']

    vec_down = read_excel(down_fname)
    speed_down = vec_down['speed'].transpose().tolist()[0]
    flow_down = vec_down['flow'].transpose().tolist()[0]
    occu_down = vec_down['occu'].transpose().tolist()[0]

    x = []
    y_speed = []
    y_flow = []
    y_occu = []
    time_series_input = {}
    time_series_output = {}

    # features = np.asarray(list(chain.from_iterable(zip(speed_up,flow_up,occu_up)))).reshape(-1,3)
    features = np.asarray(list(chain.from_iterable(zip(speed_up,flow_up)))).reshape(-1,2)


    time_steps = 100
    # seq_len = 3
    seq_len = 2

    batch_size = len(features)-time_steps

    scalers = {}
    scalers['down_speed'] = vec_down['scaler_speed']
    scalers['down_flow'] = vec_down['scaler_flow']
    scalers['down_occu'] = vec_down['scaler_occu']
    scalers['up_speed'] = vec_up['scaler_speed']
    scalers['up_flow'] = vec_up['scaler_flow']
    scalers['up_occu'] = vec_up['scaler_occu']



    for i in range(len(speed_up)):
        if i+time_steps <= len(speed_up):
            feat_vec = features[i:i+time_steps].tolist() # 上游的数据
            x.append(feat_vec)
            y_speed.append(speed_down[i+time_steps-1]) # 下游的数据
            y_flow.append(flow_down[i+time_steps-1])
            y_occu.append(occu_down[i+time_steps-1])
            if test_mode:
                time_series_input[date[i+time_steps-1]] = feat_vec
                time_series_output[date[i+time_steps-1]] = [y_speed[-1],y_flow[-1],y_occu[-1]]
        else:
            break
    x = np.asarray(x)
    y = np.asarray(list(chain.from_iterable(zip(y_speed,y_flow,y_occu)))).reshape(-1,3)

    if test_mode:
        return time_series_input, time_series_output, scalers
    else:
        return x, y, scalers

    return EOFError


if __name__ == '__main__':
    # time_series_input, time_series_output, scalers = get_LSTM_input('test_up','test_down',test_mode=True)
    # print (time_series_input)

    x, y, scalers = get_LSTM_input('train_up','train_down')
    print (x[0],y[0])
    # get_SVR_input('train_up')
