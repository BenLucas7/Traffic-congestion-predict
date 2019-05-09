import configparser
import pickle
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from utils import get_SVR_input
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def test():
    speed_x, speed_y,time, scalers = get_SVR_input('test_up','speed')
    flow_x, flow_y,time, scalers = get_SVR_input('test_up','flow')

    # Load the prediction pipeline.
    with open('pickle/svr_speed_model.pickle', "rb") as model_file:
        speed_model = pickle.load(model_file)

    with open('pickle/svr_flow_model.pickle', "rb") as model_file:
        flow_model = pickle.load(model_file)

    # Calculate test score(coefficient of determination).
    speed_score = speed_model.score(speed_x, speed_y)
    flow_score = flow_model.score(flow_x, flow_y)

    print("Test score(coefficient of determination) :",speed_score,flow_score)

    predictVelocity = []
    predictFlow = []


    for i in speed_x:
        predictVelocity.append(speed_model.predict([i]).tolist()[0])

    for i in flow_x:
        predictFlow.append(flow_model.predict([i]).tolist()[0])


    print ('speed:')
    print ('mae: ',mean_absolute_error(speed_y,predictVelocity))
    print ('r2 score: ',r2_score(speed_y,predictVelocity))
    print ('mse: ',mean_squared_error(speed_y,predictVelocity))
    print ('mape: ',MAPE(speed_y,predictVelocity))
    print ('\nflow:')
    print ('mae: ',mean_absolute_error(flow_y,predictFlow))
    print ('r2 score: ',r2_score(flow_y,predictFlow))
    print ('mse: ',mean_squared_error(flow_y,predictFlow))
    print ('mape: ',MAPE(flow_y,predictFlow))

    x = range(len(speed_x))

    last=-1

    plt.rcParams['font.family'] = ['Ping Hei']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.unicode_minus'] = False

    # font = FontProperties(fname='/Users/YogaLucas/Library/Fonts/SF-Pro-Text-Regular.otf',size=12)

    # fig=plt.figure()#figsize=(15,9.375))
    # ax1=fig.add_subplot(2,1,1)
    # ax1.plot(x[:last],speed_y[:last],color='#4285F4',label="真实值",linestyle='-')
    # ax1.plot(x[:last],predictVelocity[:last],color='#DB4437',label= "预测值",linestyle='--',marker='x')
    # ax1.set_xlabel("时间/min")#,fontproperties=font)
    # ax1.set_ylabel("速度")#,fontproperties=font)
    # # ax.set_ylim(-1,1)
    # ax1.set_title('PSO-SVR 预测结果')
    # ax1.legend(loc="best",framealpha=0.5)
    #
    # error=speed_y[:last]-predictVelocity[:last]
    # ax1=fig.add_subplot(2,1,2)
    # ax1.bar(x[:last],speed_y[:last]-predictVelocity[:last])
    # ax1.set_xlabel("时间/min")#,fontproperties=font)
    # ax1.set_ylabel("绝对误差")#,fontproperties=font)
    # ax1.set_ylim(-2,2)
    # ax1.set_title('平均速度绝对误差')

    # ax2=fig.add_subplot(2,1,1)
    # ax2.plot(x[:last],flow_y[:last],color='#F4B400',label="真实值",linestyle='-' )
    # ax2.plot(x[:last],predictFlow[:last],color='#0F9D58',label= "预测值",linestyle='--',marker='x')
    # ax2.set_xlabel("时间/min")#,fontproperties=font)
    # ax2.set_ylabel("流量")#,fontproperties=font)
    # # ax.set_ylim(-1,1)
    # ax2.set_title('PSO-SVR 预测结果')
    #
    # ax2.legend(loc="best",framealpha=0.5)
    #
    # error=flow_y[:last]-predictFlow[:last]
    # ax1=fig.add_subplot(2,1,2)
    # ax1.bar(x[:last],speed_y[:last]-predictVelocity[:last])
    # ax1.set_xlabel("时间/min")#,fontproperties=font)
    # ax1.set_ylabel("绝对误差")#,fontproperties=font)
    # ax1.set_ylim(-2,2)
    # ax1.set_title('交通流量绝对误差')
    #
    #
    # # fig.suptitle('\nSVR-PSO预测结果')
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.08,hspace =0.5)
    # plt.show()



if __name__ == "__main__":
    test()
