import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_data(file,type='speed'):
    df = pd.read_excel(file,index_col = 0)
    df.index = pd.to_datetime(df.index)

    # print (df.head(28))
    date = df.index.values

    speed_list = df['speed'].tolist()
    flow_list = df['flow'].tolist()
    occu_list = df['occu'].tolist()

    x = []
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
            time_log.append(date[i+3])

        except IndexError:
            break

    # print ('x:\n',x,'\n','speed:\n',y_speed,'\n','flow:\n',y_flow,'\n')



    if type=='speed':
        return np.asarray(x), np.asarray(y_speed),np.asarray(time_log)
    if type=='flow':
        return np.asarray(x), np.asarray(y_flow),np.asarray(time_log)


    return None

if __name__ == '__main__':
    load_data('data/test_up.xlsx','speed')
