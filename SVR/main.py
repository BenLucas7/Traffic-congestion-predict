import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils
import json
import pickle
import datetime
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from matplotlib.text import OffsetFrom
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def load_model():
    with open('pickle/svr_speed_model.pickle','rb') as f:
        svr_speed = pickle.load(f)

    with open('pickle/svr_flow_model.pickle','rb') as f:
        svr_flow = pickle.load(f)

    with open('pickle/lstm_speed.pickle','rb') as f:
        lstm_speed = pickle.load(f)

    with open('pickle/lstm_flow.pickle','rb') as f:
        lstm_flow = pickle.load(f)

    with open('pickle/scaler_speed_model.pickle','rb') as f:
        scaler_speed = pickle.load(f)

    with open('pickle/scaler_flow_model.pickle','rb') as f:
        flow_speed = pickle.load(f)

    return lstm_speed,lstm_flow,svr_speed,svr_flow,scaler_speed,flow_speed


def predict_using_svr_LSTM():
    time_series_input, time_series_output, all_scaler = utils.get_LSTM_input('test_up','test_down',True)

    lstm_speed,lstm_flow,svr_speed,svr_flow,scaler_speed,flow_speed = load_model()

    delta = np.timedelta64(5,'m')

    y_pred_speed = []
    y_pred_flow = []

    x_time = [] # 画图时候的横轴
    y_true_speed = []
    y_true_flow = []
    for time,vec in time_series_input.items():
        # 获取svr输入向量 [s-10,s-5,s,f-10,f-5,f]
        tempVec = np.asarray(vec[-3:])
        speed = tempVec[:,0].reshape(-1,1)
        flow = tempVec[:,1].reshape(-1,1)
        # 逆归一化
        speed = all_scaler['up_speed'].inverse_transform(speed).transpose().ravel()
        flow = all_scaler['up_flow'].inverse_transform(flow).transpose().ravel()

        svr_input = np.append(speed,flow).reshape(1,6)

        # 5分钟以后的速度，流量
        next_up_speed = svr_speed.predict(svr_input)
        next_up_flow = svr_flow.predict(svr_input)

        # 归一化
        next_up_speed_uniformed = all_scaler['up_speed'].transform([next_up_speed])
        next_up_flow_uniformed = all_scaler['up_flow'].transform([next_up_flow])

        # 因为是预测五分钟以后，所以时间线最开始的要删除，最后加上预测值
        lstm_input = time_series_input[time][1:]
        lstm_input.append([next_up_speed_uniformed,next_up_flow_uniformed])
        next_down_speed_uniformed = lstm_speed.predict(np.asarray(lstm_input).reshape(1,100,2))
        next_down_flow_uniformed = lstm_flow.predict(np.asarray(lstm_input).reshape(1,100,2))

        next_down_speed =\
            all_scaler['down_speed'].inverse_transform(next_down_speed_uniformed).tolist()[0][0]
        next_down_flow =\
            all_scaler['down_flow'].inverse_transform(next_down_flow_uniformed).tolist()[0][0]

        # print (time,next_down_speed,next_down_flow)

        # 5分钟后
        next = time+delta

        try:
            true_speed_uniformed = time_series_output[next][0]
            true_flow_uniformed = time_series_output[next][1]

            true_speed = all_scaler['down_speed'].inverse_transform([[true_speed_uniformed]]).tolist()[0][0]
            true_flow = all_scaler['down_flow'].inverse_transform([[true_flow_uniformed]]).tolist()[0][0]

            y_pred_speed.append(next_down_speed)
            y_pred_flow.append(next_down_flow)

            y_true_speed.append(true_speed)
            y_true_flow.append(true_flow)

            x_time.append(next)

        except KeyError:
            break


    fig = plt.figure(figsize=(15,9.375))
    ax1 = plt.subplot(211)
    ax1.plot(x_time,y_pred_speed,linestyle = '-',color = '#4285F4',label = 'predict')
    ax1.plot(x_time,y_true_speed,linestyle = '-',color = '#DB4437',label = 'true')
    ax1.set(ylabel='speed')
    ax1.set_title('Off-site traffic speed prediction')
    ax1.legend(loc='best',framealpha=0.5)

    speed_annotation = '{}{:.4f}\n{}{:.4f}\n{}{:.4f}\n{}{:.4f}'.format(
            'mae: ',
            mean_absolute_error(y_true_speed,y_pred_speed),
            'r2 score: ',
            r2_score(y_true_speed,y_pred_speed),
            'mse: ',
            mean_squared_error(y_true_speed,y_pred_speed),
            'mape: ',
            MAPE(y_true_speed,y_pred_speed)
    )
    at = AnchoredText(speed_annotation,
                      prop=dict(size=10), frameon=True,
                      loc='lower left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)


    ax2 = plt.subplot(212)
    ax2.plot(x_time,y_pred_flow,linestyle = '-',color = '#F4B400',label = 'predict')
    ax2.plot(x_time,y_true_flow,linestyle = '-',color = '#0F9D58',label = 'true')
    ax2.set(ylabel='flow')
    ax2.legend(loc='best',framealpha=0.5)
    ax2.set_title('Off-site traffic flow prediction')

    flow_annotation = '{}{:.4f}\n{}{:.4f}\n{}{:.4f}\n{}{:.4f}'.format(
            'mae: ',
            mean_absolute_error(y_true_flow,y_pred_flow),
            'r2 score: ',
            r2_score(y_true_flow,y_pred_flow),
            'mse: ',
            mean_squared_error(y_true_flow,y_pred_flow),
            'mape: ',
            MAPE(y_true_flow,y_pred_flow)
    )

    print (flow_annotation)

    at = AnchoredText(flow_annotation,
                      prop=dict(size=10), frameon=True,
                      loc='lower left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    ax2.add_artist(at)


    plt.savefig('pic/Final.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    predict_using_svr_LSTM()
