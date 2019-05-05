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
    speed_scaler = return_dict['scaler_speed']
    flow_scaler = return_dict['scaler_flow']

    speed = speed_scaler.inverse_transform(speed)
    flow = flow_scaler.inverse_transform(flow)

    _, mw,ew,ow = cal_entropy_weight(time,speed,flow)

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
    print (time[cut])
    # speed = return_dict['speed']
    # flow = return_dict['flow']

    pred_speed = np.array(pred_speed).reshape(-1,1)
    pred_flow = np.array(pred_flow).reshape(-1,1)
    true_speed = np.array(true_speed).reshape(-1,1)
    true_flow = np.array(true_flow).reshape(-1,1)

    print (len(time[cut:]),pred_speed.shape)

    return time[cut:], pred_speed, pred_flow, true_speed, true_flow
    # return time,speed[:-1],flow[:-1]


def rush_type_judge(hour):
    if 6<=hour<8: # morning_rush 6-8
        return 1
    elif 17<=hour<19: # evening_rush 17-19
        return 2
    else:
        return 0

def speed_membership(x):
    """
    fixed = [40,50,60,70,80] +-5
    """
    fixed = [40,50,60,70,80]
    loose = [ [i-5,i+5] for i in fixed]
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
    """
    fixed = [1,1.5,2,2.5,3] +-0.5
    """
    fixed = [1,1.5,2,2.5,3]
    loose = [ [i-0.5,i+0.5] for i in fixed]
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
    """
    fixed = [0.25,0.4,0.55,0.7,0.85] +- 0.2
    """
    fixed = [0.25,0.4,0.55,0.7,0.85]
    loose = [ [i-0.2,i+0.2] for i in fixed]
    k = [item for sublist in loose for item in sublist]
    score = []
    score.append(func_left(k[0],k[1],x))
    score.append(func_middle(k[0],k[1],k[2],k[3],x))
    score.append(func_middle(k[2],k[3],k[4],k[5],x))
    score.append(func_middle(k[4],k[5],k[6],k[7],x))
    score.append(func_middle(k[6],k[7],k[8],k[9],x))
    score.append(func_right(k[8],k[9],x))

    return score


def tag_rush_type(time,matrix):
    matrix = np.append(matrix,np.zeros((matrix.shape[0],1)),axis=1)
    print (time[0],time[-1])
    print (len(time),matrix.shape[0])

    for i in range(matrix.shape[0]):
        hour = pd.to_datetime(time[i]).hour
        matrix[i][-1] = rush_type_judge(hour)

    return matrix


def cal_weight(matrix):
    p = matrix / matrix.sum(axis=0)
    k = -1/np.log(p.shape[0])
    plnp = np.nan_to_num(p * np.log(p))
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

def matrix_with_rush_tag(time,speed,flow):
    speed = np.asarray(speed).reshape(-1,1)
    flow = np.asarray(flow).reshape(-1,1)
    matrix = np.append(speed,(flow/speed/12).reshape(-1,1),axis=1)
    matrix = np.append(matrix,(flow/183).reshape(-1,1),axis=1)
    matrix = tag_rush_type(time,matrix)
    return matrix


def cal_entropy_weight(time,speed,flow):
    matrix = matrix_with_rush_tag(time,speed,flow)
    matrix_uni = np.copy(matrix)
    print ('SPEED MAX',np.max(matrix_uni[:,0]),np.min(matrix_uni[:,0]))
    print ('DENSE MAX',np.max(matrix_uni[:,1]),np.min(matrix_uni[:,1]))
    print ('S MAX',np.max(matrix_uni[:,2]),np.min(matrix_uni[:,2]))

    # Speed
    matrix_uni[:,0] = ((np.max(matrix_uni[:,0]) - matrix_uni[:,0])/(np.max(matrix_uni[:,0])-np.min(matrix_uni[:,0])))
    # D
    matrix_uni[:,1] = ((matrix_uni[:,1] - np.min(matrix_uni[:,1]))/(np.max(matrix_uni[:,1])-np.min(matrix_uni[:,1])))
    # S
    matrix_uni[:,2] = ((matrix_uni[:,2] - np.min(matrix_uni[:,2]))/(np.max(matrix_uni[:,2])-np.min(matrix_uni[:,2])))

    print (matrix_uni[np.where(matrix_uni[:,-1]==0)].shape[0],
            matrix_uni[np.where(matrix_uni[:,-1]==1)].shape[0],
            matrix_uni[np.where(matrix_uni[:,-1]==2)].shape[0],
            matrix_uni[np.where(matrix_uni[:,-1]==0)].shape[0]+
                    matrix_uni[np.where(matrix_uni[:,-1]==1)].shape[0]+
                    matrix_uni[np.where(matrix_uni[:,-1]==2)].shape[0])

    mor_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==1)][:,:3])
    eve_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==2)][:,:3])
    oth_weight = cal_weight(matrix_uni[np.where(matrix_uni[:,-1]==0)][:,:3])

    print ('mor: ',mor_weight)
    print ('eve: ',eve_weight)
    print ('oth: ',oth_weight)


    return matrix,mor_weight,eve_weight,oth_weight


def congestion_rank(matrix,mor_weight,eve_weight,oth_weight):

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
    time,pred_speed,pred_flow,true_speed,true_flow = read_json()
    # time,pred_speed,pred_flow = read_json()
    mor_weight,eve_weight,oth_weight = get_final_weight()
    pred_matrix = matrix_with_rush_tag(time,pred_speed,pred_flow)
    a = congestion_rank(pred_matrix,mor_weight,eve_weight,oth_weight)
    true_matrix= matrix_with_rush_tag(time,true_speed,true_flow)
    b = congestion_rank(true_matrix,mor_weight,eve_weight,oth_weight)

    cnt=0
    for i in range(len(a)):
        if a[i]!=b[i]:
            cnt+=1

    print ((len(a)-cnt)/len(a))
