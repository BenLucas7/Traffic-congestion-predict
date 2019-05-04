from matplotlib import pyplot as plt
import json
import numpy as np
from utils import read_excel
import datetime
import pandas as pd
from bisect import bisect


def func_left(k1,k2,x):
    if x<k1:
        return 1
    elif k1<=x<k2:
        return (k2-x)/(k2-k1)
    else:
        return 0

def func_middle(k1,k2,k3,k4,x):
    if k1<=x<k2:
        return (x-k1)/(k2-k1)
    elif k2<=x<k3:
        return 1
    elif k3<=x<k4:
        return (k4-x)/(k4-k3)
    else:
        return 0

def func_right(k8,k9,x):
    if x<k8:
        return 0
    elif k8<=x<k9:
        return (x-k8)/(k9-k8)
    else:
        return 1

def get_final_weight():
    return_dict = read_excel('train_up')
    time = return_dict['time']
    speed = return_dict['speed']
    flow = return_dict['flow']

    _, mw,ew,ow = entropy_weight_method(time,speed,flow,True)

    return mw,ew,ow



def read_json():
    return_dict = read_excel('test_down')
    time = return_dict['time']
    #
    with open('y_pred_speed.json','r') as f:
        pred_speed = json.load(f)

    with open('y_pred_flow.json','r') as f:
        pred_flow = json.load(f)

    with open('y_true_speed.json','r') as f:
        true_speed = json.load(f)

    with open('y_true_flow.json','r') as f:
        true_flow = json.load(f)

    cut = np.where(time==np.datetime64('2019-04-08T08:15:00'))[0].tolist()[0] + 1
    #
    # speed = return_dict['speed']
    # flow = return_dict['flow']

    return time[cut:], pred_speed,pred_flow,true_speed,true_flow
    # return time,speed[:-1],flow[:-1]


def rush_type_judge(hour):
    if 6<=hour<8: # morning_rush 6-8
        return 1
    elif 17<=hour<19: # evening_rush 17-19
        return 2
    else:
        return 0

def speed_membership(x):
    fixed = [20,30,40,50,60]
    loose = [ [i-4,i+4] for i in fixed]
    k = [item for sublist in loose for item in sublist]
    score = []
    score.append(func_left(k[0],k[1],x))
    score.append(func_middle(k[0],k[1],k[2],k[3],x))
    score.append(func_middle(k[2],k[3],k[4],k[5],x))
    score.append(func_middle(k[4],k[5],k[6],k[7],x))
    score.append(func_middle(k[6],k[7],k[8],k[9],x))
    score.append(func_right(k[8],k[9],x))
    return score[::-1]

def dense_membership(x):
    fixed = [10,20,30,40,50]
    loose = [ [i-4,i+4] for i in fixed]
    k = [item for sublist in loose for item in sublist]
    score = []
    score.append(func_left(k[0],k[1],x))
    score.append(func_middle(k[0],k[1],k[2],k[3],x))
    score.append(func_middle(k[2],k[3],k[4],k[5],x))
    score.append(func_middle(k[4],k[5],k[6],k[7],x))
    score.append(func_middle(k[6],k[7],k[8],k[9],x))
    score.append(func_right(k[8],k[9],x))

    return score

def saturate_membership(x):
    fixed = [0.25,0.4,0.55,0.7,0.85]
    loose = [ [i-0.07,i+0.07] for i in fixed]
    k = [item for sublist in loose for item in sublist]
    score = []
    score.append(func_left(k[0],k[1],x))
    score.append(func_middle(k[0],k[1],k[2],k[3],x))
    score.append(func_middle(k[2],k[3],k[4],k[5],x))
    score.append(func_middle(k[4],k[5],k[6],k[7],x))
    score.append(func_middle(k[6],k[7],k[8],k[9],x))
    score.append(func_right(k[8],k[9],x))

    return score


def tag_rush_type(time,matrix,train=False):
    matrix = np.append(matrix,np.zeros((matrix.shape[0],1)),axis=1)
    if not train:
        time = time[:-1] # the last element is dropped for there is no true value to compared with

    print (len(time),matrix.shape[0])

    for i in range(matrix.shape[0]):
        hour = pd.to_datetime(time[i]).hour
        matrix[i][-1] = rush_type_judge(hour)

    return matrix


def cal_weight(matrix):

    p = matrix / matrix.sum(axis=0)
    k = -1/np.log(p.shape[0])
    plnp = np.nan_to_num(p * np.log(p))
    # print (np.where(np.isnan(e)))
    e = k * np.sum(plnp,axis=0)
    g = 1-e
    weight = g / g.sum()

    return np.asarray(weight).reshape(1,3)

def get_R_matrix(vec):
    r = []
    r.append(speed_membership(vec[0]))
    r.append(dense_membership(vec[1]))
    r.append(saturate_membership(vec[2]))

    return np.asarray(r)



def entropy_weight_method(time,speed,flow,train=False):
    # speed = np.asarray(speed).reshape(-1,1)
    # flow = np.asarray(flow).reshape(-1,1)
    matrix = np.append(speed,(flow*12/speed).reshape(-1,1),axis=1)
    matrix = np.append(matrix,(flow/183).reshape(-1,1),axis=1)

    # matrix [V,D,S]

    # 归一化

    matrix = tag_rush_type(time,matrix,train)
    matrix_uni = np.copy(matrix)
    matrix_uni[:,0] = ((np.max(matrix_uni[:,0]) - matrix_uni[:,0])/(np.max(matrix_uni[:,0])-np.min(matrix_uni[:,0])))
    matrix_uni[:,1] = ((matrix_uni[:,1] - np.min(matrix_uni[:,1]))/(np.max(matrix_uni[:,1])-np.min(matrix_uni[:,1])))
    matrix_uni[:,2] = ((matrix_uni[:,2] - np.min(matrix_uni[:,2]))/(np.max(matrix_uni[:,2])-np.min(matrix_uni[:,2])))
    # print (matrix[np.where(matrix[:,-1]==0)].shape[0],
    #         matrix[np.where(matrix[:,-1]==1)].shape[0],
    #         matrix[np.where(matrix[:,-1]==2)].shape[0],
    #         matrix[np.where(matrix[:,-1]==0)].shape[0]+
    #                 matrix[np.where(matrix[:,-1]==1)].shape[0]+
    #                 matrix[np.where(matrix[:,-1]==2)].shape[0])

    mor_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==1)][:,:3])
    eve_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==2)][:,:3])
    oth_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==0)][:,:3])

    print ('mor: ',mor_weight)
    print ('eve: ',eve_weight)
    print ('oth: ',oth_weight)

    return matrix,mor_weight,eve_weight,oth_weight


def congestion_rank(matrix,mor_weight,eve_weight,oth_weight):

    mor_weight = np.asarray([[0.02143989,0.96628923,0.01227088]])
    eve_weight = np.asarray([[3.66285542e-02,9.63180410e-01,1.91035855e-04]])
    oth_weight = np.asarray([[0.03645846,0.93546843,0.02807311]])
    #

    print (matrix[np.where(matrix[:,-1]==0)].shape[0],
            matrix[np.where(matrix[:,-1]==1)].shape[0],
            matrix[np.where(matrix[:,-1]==2)].shape[0],
            matrix[np.where(matrix[:,-1]==0)].shape[0]+
                    matrix[np.where(matrix[:,-1]==1)].shape[0]+
                    matrix[np.where(matrix[:,-1]==2)].shape[0])

    for i in range(matrix.shape[0]):#np.argmax(res)
        R = get_R_matrix(matrix[i][:-1])
        if matrix[i][-1]==1:
            res = mor_weight.dot(R)
            matrix[i][-1] = np.argmax(res)+1
        elif matrix[i][-1]==2:
            res = eve_weight.dot(R)
            matrix[i][-1] = np.argmax(res)+1
        else:
            res = oth_weight.dot(R)
            matrix[i][-1] = np.argmax(res)+1

    print (
            matrix[np.where(matrix[:,-1]==1)].shape[0],
            matrix[np.where(matrix[:,-1]==2)].shape[0],
            matrix[np.where(matrix[:,-1]==3)].shape[0],
            matrix[np.where(matrix[:,-1]==4)].shape[0],
            matrix[np.where(matrix[:,-1]==5)].shape[0],
            matrix[np.where(matrix[:,-1]==6)].shape[0],

            matrix[np.where(matrix[:,-1]==0)].shape[0]+
                    matrix[np.where(matrix[:,-1]==1)].shape[0]+
                    matrix[np.where(matrix[:,-1]==2)].shape[0]+
                    matrix[np.where(matrix[:,-1]==3)].shape[0]+
                    matrix[np.where(matrix[:,-1]==4)].shape[0]+
                    matrix[np.where(matrix[:,-1]==5)].shape[0]+
                    matrix[np.where(matrix[:,-1]==6)].shape[0]
                    )

    return matrix[:,-1].reshape(1,-1)[0]
    # print (np.sum(e,axis=1))

if __name__ == '__main__':
    # time,pred_speed,pred_flow,true_speed,true_flow = read_json()
    # # time,pred_speed,pred_flow = read_json()
    #
    # pred_matrix,mor_weight,eve_weight,oth_weight = entropy_weight_method(time,pred_speed,pred_flow)
    # a = congestion_rank(pred_matrix,mor_weight,eve_weight,oth_weight)
    # true_matrix,mor_weight,eve_weight,oth_weight = entropy_weight_method(time,true_speed,true_flow)
    # b = congestion_rank(true_matrix,mor_weight,eve_weight,oth_weight)
    #
    # cnt=0
    # for i in range(len(a)):
    #     if a[i]!=b[i]:
    #         cnt+=1
    #
    # print ((len(a)-cnt)/len(a))

    get_final_weight()
