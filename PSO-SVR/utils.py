import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import MinMaxScaler

palette = {
    'blue':{'d':'#096097','l':'#98AFCF'},
    'skyblue':{'d':'#1A99D7','l':'#A0D1E2'},
    'yellow':{'d':'#EBA328','l':'#F8D293'},
    'green':{'d':'#158684','l':'#96C6CE'},
    'purple':{'d':'#A25E7E','l':'#B6949F'},
    'red':{'d':'#DE6A69','l':'#EDB9B3'}
}

themeSet = {
    # 渐变色
    'gradient':{'gru_attn':{'color':'#0D427F','marker':''},
                'gru':{'color':'#146AAA','marker':''},
                'attn':{'color':'#328DBC','marker':''},
                'gru4rec':{'color':'#53B4D2','marker':''},
                'sasrec':{'color':'#7ECCC4','marker':''},
                'deems':{'color':'#CDEBC7','marker':''},
                'din':{'color':'#E1F3DC','marker':''},
                'dream':{'color':'#F7FCF1','marker':''},
    },

    # 彩色
    'coloful':{'gru_attn':{'color':'#EA616A','marker':''},
                'gru':{'color':'#F7915D','marker':''},
                'attn':{'color':'#9C9184','marker':''},
                'gru4rec':{'color':'#9AC696','marker':''},
                'sasrec':{'color':'#63B3B3','marker':''},
                'deems':{'color':'#689ACA','marker':''},
                'din':{'color':'#C496C4','marker':''},
                'dream':{'color':'#AA7969','marker':''},
    },

    # 糖果色
    'candy':{'gru_attn':{'color':'#5ABAA7','marker':''},
                'gru':{'color':'#99D5CA','marker':''},
                'attn':{'color':'#328DBC','marker':''},
                'gru4rec':{'color':'#FFC374','marker':''},
                'sasrec':{'color':'#D1BCD2','marker':''},
                'deems':{'color':'#FDB3B3','marker':''},
                'din':{'color':'#9C9184','marker':''},
                'dream':{'color':'#C9C9C7','marker':''},
    }
}

metricSet = {
    'AUC':{'name':'AUC','index':0},
    'P':{'name':'P','index':1},
    'R':{'name':'R','index':2},
    'NDCG':{'name':'NDCG','index':3},
    'GP':{'name':'GP','index':4},
    'GAUC':{'name':'GAUC','index':5},
    'F1':{'name':'F1','index':6},
    'MAP':{'name':'MAP','index':7},
    'MRR':{'name':'MRR','index':8},
}

def get_SVR_input(fname,time_window=3,scale=True):
    print(scale)
    df = pd.read_excel('data/'+fname+'.xlsx',index_col = 0)
    df.drop(columns=['occu'],inplace=True)
    df.index = pd.to_datetime(df.index)

    time = df.index.values.reshape(-1)
    speed = df.speed.values.reshape(-1)
    flow = df.flow.values.reshape(-1)


    x = []
    y_speed = []
    y_flow = []
    time_log = []

    for i in range(0,speed.shape[0],time_window+1):
        try:
            x.append(np.r_[speed[i:i+time_window],flow[i:i+time_window]])
            y_speed.append(speed[i+time_window])
            y_flow.append(flow[i+time_window])
            time_log.append(time[i+time_window]) # just for plot

        except IndexError:
            break

    if scale:
        speed_scaler = MinMaxScaler()
        speed_scaler.fit(speed.reshape(-1,1))
        flow_scaler = MinMaxScaler()
        flow_scaler.fit(flow.reshape(-1,1))

        #speed scale
        x = np.array(x)
        x[:,:time_window] = speed_scaler.transform(x[:,:time_window])
        #flow scale
        x[:,time_window:time_window*2] = flow_scaler.transform(x[:,time_window:time_window*2])
        return np.c_[x,y_speed,y_flow],speed_scaler,flow_scaler

    return np.c_[x,y_speed,y_flow],None,None

if __name__ == '__main__':
    # get_LSTM_input('train',True)
    mat,_,s = get_SVR_input('train',5,False)
    print(mat.shape,mat)
