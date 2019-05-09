import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import json
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Ping Hei']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Ellipse
import pandas as pd
from matplotlib.text import OffsetFrom
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib.patches as patches
# use image magick (this is bash, not python)
# convert -delay 80 *.png animated_chart.gif

#speed cmax = 1400 cmin=0 cmean=321,
# gmax=2,gmin=0 gmean = 0.016 fit gamma[0-0.03],c[200-400]

def getPlot(train_type):

    with open('c_' + train_type + '.json','r') as f:
        c_pair_plot = json.load(f)
    with open('gamma_'+ train_type + '.json','r') as f:
        gamma_pair_plot = json.load(f)

    with open(train_type + '_best_params.json','r') as f:
        best_params = json.load(f)

    num = len(gamma_pair_plot)


    ctemp = np.asarray(c_pair_plot).ravel()
    print (max(ctemp),min(ctemp),np.mean(ctemp))
    gtemp = np.asarray(gamma_pair_plot).ravel()
    print (max(gtemp),min(gtemp),np.mean(gtemp))


    # plt.subplot(121)
    # plt.hist(ctemp)
    # plt.subplot(122)
    # plt.hist(gtemp)
    # plt.show()
    cmean = np.mean(ctemp)
    gmean = np.mean(gtemp)

    # pairs=list(zip(c_pair_plot,gamma_pair_plot))
    # print(pairs)
    #
    # cnt=0
    # scat = ax.scatter([], [], s=2)
    #
    # def update(i):
    #     scat.set_offsets(pairs)
    #     return scat,
    #
    # anim = animation.FuncAnimation(fig, update, interval=25, blit=False)
    # plt.show()


    for i in range(num):
        print (i)
        # fig = plt.figure()
        # plt.subplot(111)
        # plt.scatter(c_pair_plot[i],gamma_pair_plot[i],s=3)
        # # plt.xlim(0,30)
        # # plt.ylim(0,0.1)
        # plt.xlabel('C')
        # plt.ylabel('gamma')
        # plt.title('PSO Optimization for c and gamma')
        # filename='step'+str(i)+'.png'
        # plt.savefig('plot_' + train_type + '/'+filename, dpi=300)
        # plt.gca()

        fig = plt.figure(figsize=(15,9.375))
        ax1 = plt.subplot(121)
        ax1.scatter(c_pair_plot[i],gamma_pair_plot[i],s=3)
        #------------- speed ---------------#
        ax1.set_xlim(0,400)
        ax1.set_ylim(0,0.05)
        rect = patches.Rectangle((300,0.0012),75,0.0005,linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(rect)
        #------------- speed ---------------#

        # #------------- flow  ---------------#
        # # ax1.set_xlim(0,5000)
        # # ax1.set_ylim(0,0.02)
        # rect = patches.Rectangle((2000,0.0090),200,0.0009,linewidth=1,edgecolor='r',facecolor='none')
        # # Add the patch to the Axes
        # ax1.add_patch(rect)
        # #------------- flow  ---------------#

        ax1.set(xlabel='C', ylabel='gamma')

        ax2 = plt.subplot(122)
        ax2.scatter(c_pair_plot[i],gamma_pair_plot[i],s=3)
        # ------------- speed ---------------#
        ax2.set_xlim(300,375)
        ax2.set_ylim(0.0012,0.0017)
        # ------------- speed ---------------#

        # #------------- flow  ---------------#
        # ax2.set_xlim(2000,2200)
        # ax2.set_ylim(0.0090,0.0099)
        # #------------- flow  ---------------#
        ax2.set(xlabel='C', ylabel='gamma')
        at = AnchoredText("Zoomed in",
                          prop=dict(size=15), frameon=True,
                          loc='upper right',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at)

        fig.suptitle('\nPSO Optimization for c and gamma')

        if i == num-1:
            # ------------- speed ---------------#
            ax2.annotate('global best', xy=(311.515, 0.00132491), xytext=(315, 0.00133),
                arrowprops=dict(arrowstyle="->"),
                )
            # ------------- speed ---------------#

            # #------------- flow  ---------------#
            # ax2.annotate('global best', xy=(2091.37174247, 0.009577), xytext=(2110, 0.00955),
            #     arrowprops=dict(arrowstyle="->"),
            #     )
            # #------------- flow  ---------------#

        filename='step'+str(i)+'.png'
        plt.savefig('plot_' + train_type + '/'+filename, dpi=300)
        plt.gca()


def get_speed_flow_together():

    with open('json/c_speed.json','r') as f:
        speed_c = json.load(f)
    with open('json/gamma_speed.json','r') as f:
        speed_g = json.load(f)
    with open('json/speed_best_params.json','r') as f:
        speed_best = json.load(f)

    with open('json/c_flow.json','r') as f:
        flow_c = json.load(f)
    with open('json/gamma_flow.json','r') as f:
        flow_g = json.load(f)
    with open('json/flow_best_params.json','r') as f:
        flow_best = json.load(f)

    num = len(speed_c)

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.scatter(speed_c[-1],speed_g[-1],s=4)
    ax1.set(xlabel='speed_C', ylabel='speed_gamma')
    ax1.set_title('粒子群优化速度预测模型时参数的变化')
    ax1.annotate('global best', xy=(speed_best['C'], speed_best['gamma']), xytext=(speed_best['C']+speed_best['gamma'], speed_best['gamma']*1.005),
        arrowprops=dict(arrowstyle="->"),
    )

    ax2 = plt.subplot(122)
    ax2.scatter(flow_c[-1],flow_g[-1],s=4)
    ax2.set(xlabel='flow_C', ylabel='flow_gamma')
    ax2.set_title('粒子群优化流量预测模型时参数的变化')
    ax2.annotate('global best', xy=(flow_best['C'], flow_best['gamma']), xytext=(flow_best['C']+flow_best['gamma']
                                                                                    , flow_best['gamma']*1.005),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.show()

if __name__ == '__main__':
    get_speed_flow_together()
