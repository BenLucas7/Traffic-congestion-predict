import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_data(type):
    df = pd.read_excel('data/April.xlsx',index_col = 0)
    df.index = pd.to_datetime(df.index)

    # print (df.head(28))
    speed_list = df['speed'].tolist()
    flow_list = df['flow'].tolist()
    occu_list = df['occu'].tolist()

    x = []
    y_speed = []
    y_flow = []


    for i in range(0,len(speed_list),4):
        try:
            vec = [speed_list[i],speed_list[i+1],speed_list[i+2]]
            vec.extend([flow_list[i],flow_list[i+1],flow_list[i+2]])
            x.append(vec)
            y_speed.append(speed_list[i+3])
            y_flow.append(flow_list[i+3])

        except IndexError:
            break

    # print ('x:\n',x,'\n','speed:\n',y_speed,'\n','flow:\n',y_flow,'\n')

    if type=='speed':
        return np.asarray(x), np.asarray(y_speed)
    if type=='flow':
        return np.asarray(x), np.asarray(y_flow)

    return None
