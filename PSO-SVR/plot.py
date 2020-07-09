import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import palette,metricSet

import argparse
import os

def draw_scatter():
    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'evaluation',args.type)
    print(path)
    mat = np.load(path+'/{}_param_pair.npy'.format(args.type))
    df1 = pd.DataFrame(mat[:100,:],columns=['c','gamma','score'])
    df2 = pd.DataFrame(mat[6200:6300,:],columns=['c','gamma','score'])
    df3 = pd.DataFrame(mat[18600:18700,:],columns=['c','gamma','score'])
    df4 = pd.DataFrame(mat[24900:25000,:],columns=['c','gamma','score'])
    # print(mat.shape)
    figure,axes = plt.subplots(nrows=1,ncols=4,figsize=(20,4))
    df1.plot(ax=axes[0],kind="scatter", x="c", y="gamma", alpha=0.5, c="score", cmap=plt.get_cmap("jet"), colorbar=False,)
    df2.plot(ax=axes[1],kind="scatter", x="c", y="gamma", alpha=0.5, c="score", cmap=plt.get_cmap("jet"), colorbar=False,)
    df3.plot(ax=axes[2],kind="scatter", x="c", y="gamma", alpha=0.5, c="score", cmap=plt.get_cmap("jet"), colorbar=False,)
    df4.plot(ax=axes[3],kind="scatter", x="c", y="gamma", alpha=0.5, c="score", cmap=plt.get_cmap("jet"), colorbar=False,)
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pic')
    os.makedirs(path,exist_ok=True)
    plt.savefig(path+'/{}_scatter.png'.format(args.type),dpi=500)

def draw_curve():
    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'evaluation',args.type)

    mat = np.load(path+'/{}_evaluation.npy'.format(args.type))
    df = pd.DataFrame(mat,columns=['mae','r2','mse','mape'])
    df['idx'] = df.index
    figure,axes = plt.subplots(nrows=1,ncols=4,figsize=(20,4))
    df.plot(ax=axes[0],kind='line',x='idx',y='mae')
    df.plot(ax=axes[1],kind='line',x='idx',y='r2')
    df.plot(ax=axes[2],kind='line',x='idx',y='mse')
    df.plot(ax=axes[3],kind='line',x='idx',y='mape')
    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pic')
    os.makedirs(path,exist_ok=True)
    plt.savefig(path + '/{}_curve.png'.format(args.type),dpi=500)

def draw_bar():
    speed=np.array([[0.375,0.103,0.096,0.134,0.256],# mse
        [0.476,0.267,0.246,0.335,0.350]])# mape

    flow=np.array([[8.731,5.268,3.890,7.432,6.908],# mse
        [2.743,2.986,1.715,2.307,2.675]])# mape

    name_list = ['1','2','3','4','5']
    color_list = [palette['skyblue']['d'],palette['skyblue']['l'],
                  palette['red']['d'],palette['red']['l'],
                  palette['yellow']['d'],palette['yellow']['l']]

    def addp(position):
        last = position % 10
        if last == 9:
            position = (position//10)*100+last+1
        else:
            position+=1
        return position


    fig = plt.figure(figsize=(16,4),dpi=150)

    def make_square_axes(ax):
        ax.set_aspect(1 / ax.get_data_ratio())


    width = 0.6
    x = [0.6,1.4,2.2,3.0,3.8]
    position=141
    plotTitle=True

    ax1 = plt.subplot(position)
    ax1.bar(x[0],speed[0][0],fc=palette['blue']['d'],label='1')
    ax1.bar(x[1],speed[0][1],fc=palette['skyblue']['d'],label='2')
    ax1.bar(x[2],speed[0][2],fc=palette['green']['d'],label='3')
    ax1.bar(x[3],speed[0][3],fc=palette['green']['l'],label='4')
    ax1.bar(x[4],speed[0][4],fc=palette['yellow']['d'],label='5')
    # ax1.set_ylim(0.78,0.9)
    if plotTitle:
        ax1.set_title('Speed',fontsize=14,fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_ylabel('MSE',fontsize=14,fontweight='bold')
    ax1.set_xticklabels(['1','2', '3', '4', '5'], fontsize='large')
    make_square_axes(plt.gca())
    position = addp(position)

    ax2 = plt.subplot(position)
    ax2.bar(x[0],speed[1][0],fc=palette['blue']['d'],label='1')
    ax2.bar(x[1],speed[1][1],fc=palette['skyblue']['d'],label='2')
    ax2.bar(x[2],speed[1][2],fc=palette['green']['d'],label='3')
    ax2.bar(x[3],speed[1][3],fc=palette['green']['l'],label='4')
    ax2.bar(x[4],speed[1][4],fc=palette['yellow']['d'],label='5')
    # ax1.set_ylim(0.78,0.9)
    if plotTitle:
        ax2.set_title('Speed',fontsize=14,fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_ylabel('MAPE',fontsize=14,fontweight='bold')
    ax2.set_xticklabels(['1','2', '3', '4', '5'], fontsize='large')
    make_square_axes(plt.gca())

    position = addp(position)

    ax3 = plt.subplot(position)
    ax3.bar(x[0],flow[0][0],fc=palette['blue']['d'],label='1')
    ax3.bar(x[1],flow[0][1],fc=palette['skyblue']['d'],label='2')
    ax3.bar(x[2],flow[0][2],fc=palette['green']['d'],label='3')
    ax3.bar(x[3],flow[0][3],fc=palette['green']['l'],label='4')
    ax3.bar(x[4],flow[0][4],fc=palette['yellow']['d'],label='5')
    # ax1.set_ylim(0.78,0.9)
    if plotTitle:
        ax3.set_title('Flow',fontsize=14,fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_ylabel('MSE',fontsize=14,fontweight='bold')
    ax3.set_xticklabels(['1','2', '3', '4', '5'], fontsize='large')
    make_square_axes(plt.gca())

    position = addp(position)

    ax4 = plt.subplot(position)
    ax4.bar(x[0],flow[1][0],fc=palette['blue']['d'],label='1')
    ax4.bar(x[1],flow[1][1],fc=palette['skyblue']['d'],label='2')
    ax4.bar(x[2],flow[1][2],fc=palette['green']['d'],label='3')
    ax4.bar(x[3],flow[1][3],fc=palette['green']['l'],label='4')
    ax4.bar(x[4],flow[1][4],fc=palette['yellow']['d'],label='5')
    # ax1.set_ylim(0.78,0.9)
    if plotTitle:
        ax4.set_title('Flow',fontsize=14,fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_ylabel('MSE',fontsize=14,fontweight='bold')
    ax4.set_xticklabels(['1','2', '3', '4', '5'], fontsize='large')
    make_square_axes(plt.gca())

    plt.tight_layout()
    plt.savefig('bar.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',required=True,type=str)
    parser.add_argument('--win',required=True,type=int)
    parser.add_argument('--kind',required=True,type=str)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=True)
    args = parser.parse_args()

    if args.kind=='s':
        draw_scatter()
    elif args.kind=='c':
        draw_curve()
    else:
        draw_bar()
