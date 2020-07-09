import configparser
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import json
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from utils import get_SVR_input
import time
import argparse


param_pair_plot = []
evaluation = []

def MAPE(true,pred):

    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100
#---------------------------PSO_SVR class-------------------------------------#

class PSO_SVR:
    """
    在给定数据集上训练一组SVR, 获取 SVR 的最佳超参数 C 和 gamma
    """

    def __init__(self, args):

        args = parser.parse_args()
        type = args.type
        scale = args.scale
        # PSO hyperparams.
        self.n_iterations       = args.n_iterations
        self.inertia_wt         = args.inertia_weight
        self.c1                 = args.c1
        self.c2                 = args.c2

        # validation hyperparams.
        self.validation_type    = args.validation_type
        self.n_validations      = args.validation_cv
        self.validation_size    = args.validation_size

        # SVR hyperparams.
        self.time_window        = args.win
        self.kernel             = args.kernel
        self.epsilon            = args.epsilon
        self.C_min              = args.c_min
        self.C_max              = args.c_max
        self.C_step             = args.c_step
        self.gamma_min          = args.gamma_min
        self.gamma_max          = args.gamma_max
        self.gamma_step         = args.gamma_step

        # input
        print('train scale',scale)
        mat,speed_scaler,flow_scaler = get_SVR_input(fname='train',
                                                     time_window=self.time_window,
                                                     scale=scale)
        self.X = mat[:,:self.time_window]
        if type=='speed':
            self.Y = mat[:,-2]
        elif type=='flow':
            self.Y = mat[:,-1]
        else:
            raise IndexError

        # initialize particles
        self.init_particles()
        print("Number of SVRs :", self.n_svrs)

    def init_particles(self):
        """
        每个 SVR 就是一个粒子
        初始化所有粒子参数，局部最优和全局最优。
        对于 SVR 初始化，C值从C_min（包括）到 C_max（不包括），步长为 C_step，
        对于 gamma 初始化，从 eps_min（包括）到 eps_max（不包括），步长为 eps_step
        经过如上初始化，所有可能的（C，gamma）均已考虑
        """
        # all the SVR particles.
        self.svrs = []
        # local best score of each SVR particle
        self.p_best_score = []
        # the parameters of each SVR particle when reach local best
        self.p_best_params = []
        # global best score of all SVR particle
        self.g_best_score = 0
        # the parameters of all SVR particle when reach global best
        self.g_best_params = {}

        for C in np.arange(self.C_min, self.C_max, self.C_step):
            for gamma in np.arange(self.gamma_min, self.gamma_max, self.gamma_step):
                svr = SVR(kernel = self.kernel,
                          C = C,
                          epsilon = self.epsilon,
                          gamma = gamma)

                self.svrs.append(svr)
                self.p_best_score.append(0)
                self.p_best_params.append({"C":0, "gamma": 0})

        self.n_svrs = len(self.svrs)


    def get_train_val_data(self, validation_step):
        """
        1. 将 data 划分为训练集与测试集
        2. 将 x_train 标准化为标准正态分布
        """

        if self.validation_type == "random-split":
            x_train, x_test, y_train, y_test = train_test_split(
                        self.X, self.Y, test_size = self.validation_size)
        elif self.validation_type == "k-fold":
            k = self.n_validations
            n_samples = self.Y.shape[0]
            subset_size = np.ceil(n_samples/k)

            idx_min = int((validation_step - 1) * subset_size)
            idx_max = validation_step * subset_size
            idx_max = int(min(idx_max, n_samples))

            x_test = self.X[idx_min:idx_max]
            y_test = self.Y[idx_min:idx_max]

            x_train = self.X[np.r_[0:idx_min, idx_max:n_samples]]
            y_train = self.Y[np.r_[0:idx_min, idx_max:n_samples]]


        return x_train, x_test, y_train, y_test

    def run_optimizer(self):

        start = time.time()

        for i in range(self.n_iterations):
            print("\nOptimizer step: ", i+1)

            # train all the SVRs
            self.train_svrs()
            self.update_particle_state()
            print ('training takes {:.2f} minutes'.format((time.time()-start)/60))
            print("Step best score: ", max(self.val_score))
            print("Global best score: ", self.g_best_score)
            self.evaluate()


    def train_svrs(self):
        val_score_list = [[] for i in range(self.n_svrs)]

        for val_step in range(self.n_validations):
            print ("val_step:",val_step + 1)
            x_train, x_test, y_train, y_test = self.get_train_val_data(
                        validation_step = val_step + 1)

            # train each SVR particle
            for j in range(self.n_svrs):
                self.svrs[j].fit(x_train, y_train.ravel())
                val_score = self.svrs[j].score(x_test, y_test)
                val_score_list[j].append(val_score)
                params = self.svrs[j].get_params()
                param_pair_plot.append([params['C'],params['gamma'],val_score])

        self.val_score = np.mean(val_score_list, axis = 1)


    def update_particle_state(self):

        for i in range(self.n_svrs):
            # update p_best
            if self.val_score[i] > self.p_best_score[i]:
                params = self.svrs[i].get_params()
                self.p_best_score[i] = self.val_score[i]
                self.p_best_params[i] = {'C': params['C'],
                            'gamma': params['gamma']}

                # update g_best
                if self.p_best_score[i] > self.g_best_score:
                    self.g_best_score = self.p_best_score[i]
                    self.g_best_params = params

        ctemp=[]
        gammatemp=[]

        for i in range(self.n_svrs):
            params = self.svrs[i].get_params()

            r1 = np.random.random()
            r2 = np.random.random()

            C = params['C']
            C_new = (self.inertia_wt * C \
                    + r1*self.c1*(self.p_best_params[i]['C'] - C) \
                    + r2*self.c2*(self.g_best_params['C'] - C))
            C_new = max(0.001, C_new)

            # Find new gamma value.
            gamma = params["gamma"]
            gamma_new = (self.inertia_wt * gamma \
                    + r1*self.c1*(self.p_best_params[i]["gamma"] - gamma) \
                    + r2*self.c2*(self.g_best_params["gamma"] - gamma))
            gamma_new= max(0.001, gamma_new)

            # 更新 SVR 的参数.
            self.svrs[i].set_params(**{'C': C_new, 'gamma': gamma_new})


    def get_best_values(self):
        return self.g_best_score, self.g_best_params

    def evaluate(self):
        model = SVR(**self.g_best_params)
        model.fit(self.X,self.Y)
        pred = model.predict(self.X)
        label = self.Y
        mae = mean_absolute_error(label,pred)
        r2 = r2_score(label,pred)
        mse = mean_squared_error(label,pred)
        mape = MAPE(label,pred)
        evaluation.append([mae,r2,mse,mape])




def train_svr(args,svr_params):
    args = parser.parse_args()
    type = args.type
    mat,speed_scaler,flow_scaler = get_SVR_input('train',time_window=args.win,scale=args.scale)

    X = mat[:,:args.win*2]
    if type=='speed':
        Y = mat[:,-2]
    elif type=='flow':
        Y = mat[:,-1]
    else:
        raise IndexError

    svr = SVR(**svr_params)

    pipeline = Pipeline([
        ('SVR', svr)
    ])

    pipeline.fit(X,Y)

    # Save the pipeline.
    if type=='speed':
        path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                        'pickle','speed')
        print(path)
        os.makedirs(path,exist_ok=True)
        with open(path+'/speed_svr.pickle', 'wb') as save_file:
            pickle.dump(pipeline, save_file)

    if type=='flow':
        path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                        'pickle','flow')
        print(path)
        os.makedirs(path,exist_ok=True)
        with open(path+'/flow_svr.pickle', 'wb') as save_file:
            pickle.dump(pipeline, save_file)


    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','speed')
    os.makedirs(path,exist_ok=True)
    with open(path+'/speed_scaler.pickle', 'wb') as save_file:
        pickle.dump(speed_scaler, save_file)

    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'pickle','flow')
    os.makedirs(path,exist_ok=True)
    with open(path+'/flow_scaler.pickle', 'wb') as save_file:
        pickle.dump(flow_scaler, save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type',required=True,default='speed',type=str)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=True)
    #PSO
    parser.add_argument('--n_iterations',default=30,type=int)
    parser.add_argument('--inertia_weight',default=0.729,type=float)
    parser.add_argument('--c1',default=2.05,type=float)
    parser.add_argument('--c2',default=2.05,type=float)
    #SVR
    parser.add_argument('--win',default=3,type=int)
    parser.add_argument('--kernel',default='rbf',type=str)
    parser.add_argument('--epsilon',default=0.1,type=float)
    parser.add_argument('--c_min',default=1,type=float)
    parser.add_argument('--c_max',default=100,type=float)
    parser.add_argument('--c_step',default=10,type=float)
    parser.add_argument('--gamma_min',default=0.01,type=float)
    parser.add_argument('--gamma_max',default=2,type=float)
    parser.add_argument('--gamma_step',default=0.2,type=float)
    #validation
    parser.add_argument('--validation_type',default='k-fold',type=str)
    parser.add_argument('--validation_cv',default=5,type=int)
    parser.add_argument('--validation_size',default=0.2,type=float)



    args = parser.parse_args()
    # Use PSO to find best hyper parameters for SVR.
    print("Running PSO")
    pso_optimizer = PSO_SVR(args=args)

    start = time.time()
    pso_optimizer.run_optimizer()
    print ('The whole training takes {:.2f} minutes'.format((time.time()-start)/60))

    best_score, best_params = pso_optimizer.get_best_values()
    print("\nBest score: ", best_score)
    print("Params of best SVR : ", best_params)

    path = os.path.join('log','win={} {}'.format(args.win,'scaled' if args.scale else '').rstrip(),
                    'evaluation',args.type)
    os.makedirs(path,exist_ok=True)
    with open(path+'/{}_best_params.json'.format(args.type),'w') as f:
        json.dump(best_params,f)

    # Use the best parameters to train an SVR using the whole data.
    print("\nTraining final model")
    train_svr(args,best_params)

    # parameters (c,gamma) pair log
    param_pair_plot = np.array(param_pair_plot)
    np.save(path+'/{}_param_pair.npy'.format(args.type),param_pair_plot)
    # evaluation of the global best particle log
    evaluation = np.array(evaluation)
    np.save(path+'/{}_evaluation.npy'.format(args.type),evaluation)


# TODO: 1. 画图圆圈面基表示mse，颜色表示r2 2. 画各个metric曲线图
