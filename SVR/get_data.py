import numpy as np
import pandas as pd
from itertools import chain
import json

def read_file(name):
    if name[-6]=='f':
        df = pd.read_excel('data/'+ name,index_col = 0)
        df.index = pd.to_datetime(df.index)

        df.drop(['# Lane Points','% Observed'],axis=1,inplace=True)
        # df.index.names = ['time']
        df.rename(columns = {'Lane 1 Speed (mph)':'speed', 'Lane 1 Flow (Veh/5 Minutes)':'flow'},inplace = True)
        df['time']=df.index
        # df['speed'] = df['speed'].astype('float')
        # print (df)
        # print (df[df['speed']==0.0],df[df['flow']==0.0],'\n')


    elif name[-6]=='o':
        df = pd.read_excel('data/'+ name,index_col = 0)
        df.index = pd.to_datetime(df.index)

        df.drop(['# Lane Points','% Observed'],axis=1,inplace=True)
        # df.index.names = ['time']
        df['time']=df.index
        df.rename(columns = {'Lane 1 Speed (mph)':'speed', 'Lane 1 Occ (%)':'occu'},inplace = True)
        # df['speed'] = df['speed'].astype('float')

        # print (df)

        # print (df[df['speed']==0.0],df[df['occu']==0.0],'\n')

    return df


def gen_input():
    files = []
    file_dict = {}

    for i in range(1,4):
        for j in ['up','down']:
            for k in ['sf','so']:
                name = 'week'+str(i)+'_'+j+'_'+k+'.xlsx'
                files.append(name)

    for i in range(0,len(files),2):
        try:
            name = files[i][:8]
            print (name)
            df1 = read_file(files[i])
            df2 = read_file(files[i+1])
            df3 = pd.merge_asof(df1,df2,on='time')#,by='speed')
            df3.drop(['time','speed_y'],axis=1,inplace=True)
            df3.rename(columns = {'speed_x':'speed'},inplace=True)
            file_dict[name]=df3
        except IndexError:
            break


    up = [file_dict['week1_up'],file_dict['week2_up'],file_dict['week3_up']]
    down = [file_dict['week1_do'],file_dict['week2_do'],file_dict['week3_do']]

    ups = pd.concat(up)
    downs = pd.concat(down)

    ups_speed = ups['speed'].tolist()
    ups_flow = ups['flow'].tolist()
    ups_occu = ups['occu'].tolist()

    sample_num = int(np.floor(len(ups_speed)*3/900)*900)

    ups = (np.array(list(\
            chain.from_iterable(zip(ups_speed,ups_flow,ups_occu))))).reshape((len(ups_speed),3))

    downs_speed = downs['speed'].tolist()
    downs_flow = downs['flow'].tolist()
    downs_occu = downs['occu'].tolist()
    downs = (np.array(list(\
            chain.from_iterable(zip(downs_speed,downs_flow,downs_occu))))).reshape((len(downs_speed),3))

    # for i in range(min(len(ups_speed),len(downs_speed))*3):
    #     try:
    #         temp = []
    #
    #
    #     except IndexError:
    #         break

    step=100 #切片不包括最后一个，[0,99][100,199]
    x = [ ups[i:i+step,:].tolist() for i in range(0,len(ups),step)][:-1]
    y = [ downs[i,:].tolist() for i in range(step-1,len(downs),step)]

    with open('lstm_x.json','w') as f:
        json.dump(x,f)

    with open('lstm_y.json','w') as f:
        json.dump(y,f)

    print (np.array(x).shape)
    print (np.array(y).shape)


if __name__ == '__main__':
    gen_input()

    #input 60,100,3
    #output 60,3
