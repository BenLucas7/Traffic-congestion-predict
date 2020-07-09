import configparser
import pickle
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from utils import get_SVR_input
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_squared_error
import argparse
import os

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def test(args):

    # Load the prediction pipeline.
    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','speed')
    with open(path+'/speed_svr.pickle', 'rb') as fin:
        speed_model = pickle.load(fin)

    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','flow')
    with open(path+'/flow_svr.pickle', 'rb') as fin:
        flow_model = pickle.load(fin)

    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','flow')
    with open(path+'/flow_scaler.pickle', 'rb') as fin:
        flow_scaler = pickle.load(fin)

    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','speed')
    with open(path+'/speed_scaler.pickle', 'rb') as fin:
        speed_scaler = pickle.load(fin)

    mat,_,s = get_SVR_input('test',args.win)
    speed_x = mat[:,:args.win*2]
    speed_y = mat[:,-2]
    flow_x = speed_x
    flow_y = mat[:,-1]

    speed_score = speed_model.score(speed_x, speed_y)
    flow_score = flow_model.score(flow_x, flow_y)

    print("Test score(coefficient of determination) :",speed_score,flow_score)

    speed_pred = speed_model.predict(speed_x)
    flow_pred = flow_model.predict(flow_x)

    print ('PSO')
    print ('\tspeed:')
    print ('\t\tmae: ',mean_absolute_error(speed_y,speed_pred))
    print ('\t\tr2 score: ',r2_score(speed_y,speed_pred))
    print ('\t\tmse: ',mean_squared_error(speed_y,speed_pred))
    print ('\t\tmape: ',MAPE(speed_y,speed_pred))
    print ('\tflow:')
    print ('\t\tmae: ',mean_absolute_error(flow_y,flow_pred))
    print ('\t\tr2 score: ',r2_score(flow_y,flow_pred))
    print ('\t\tmse: ',mean_squared_error(flow_y,flow_pred))
    print ('\t\tmape: ',MAPE(flow_y,flow_pred))

    # x = range(len(speed_x))

    # last=-1
    #
    # # plt.rcParams['font.family'] = ['Ping Hei']
    # # plt.rcParams['font.size'] = 15
    # # plt.rcParams['axes.unicode_minus'] = False
    # # font = FontProperties(fname='/Users/YogaLucas/Library/Fonts/SF-Pro-Text-Regular.otf',size=12)
    #
    # fig=plt.figure(dpi=200)#figsize=(15,9.375))
    # ax1=fig.add_subplot(2,1,1)
    # ax1.plot(x[:last],speed_y[:last],color='#3AB8E8',label="True",linestyle='-',lw=1.5)
    # ax1.plot(x[:last],predictVelocity[:last],color='#EE6B72',label= "Optimized",linestyle='-',lw=1.5)
    # ax1.plot(x[:last],predictVelocityPlain[:last],color='#9C9184',label= "Raw",linestyle='-',lw=1.5)
    #
    # ax1.set_xlabel("time/min")#,fontproperties=font)
    # ax1.set_ylabel("speed")#,fontproperties=font)
    # # ax.set_ylim(-1,1)
    # # ax1.set_title('PSO-SVR 预测结果')
    # ax1.legend(loc="best",framealpha=0.5)
    #
    # ax3=fig.add_subplot(2,1,2)
    # ax3.plot(x[:last],flow_y[:last],color='#3AB8E8',label="True",linestyle='-',lw=1.5)
    # ax3.plot(x[:last],predictFlow[:last],color='#EE6B72',label= "Optimized",linestyle='-',lw=1.5)
    # ax3.plot(x[:last],predictFlowPlain[:last],color='#9C9184',label= "Raw",linestyle='-',lw=1.5)
    #
    # ax3.set_xlabel("time/min")#,fontproperties=font)
    # ax3.set_ylabel("volume")#,fontproperties=font)
    # # ax.set_ylim(-1,1)
    # # ax3.set_title('PSO-SVR 预测结果')
    #
    # ax3.legend(loc="best",framealpha=0.5)
    #
    #
    #
    # # fig.suptitle('\nSVR-PSO预测结果')
    # plt.tight_layout()
    # # plt.subplots_adjust(left=0.08,hspace =0.5)
    # # plt.show()
    # plt.savefig('pic/p/res.png')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--win',required=True,type=int)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=True)
    args = parser.parse_args()
    test(args)
