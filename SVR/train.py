import configparser
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from utils import load_data
import time
c_pair_plot=[]
gamma_pair_plot=[]
#---------------------------PSO_SVR class-------------------------------------#

class PSO_SVR:
    """
    在给定数据集上训练一组SVR, 获取 SVR 的最佳超参数 C 和 gamma
    """

    def __init__(self, config_file, type):
        # 读取配置文件.
        config = configparser.ConfigParser()
        config.read(config_file)

        # PSO 参数.
        pso_config              = config["PSO"]
        self.n_iterations       = int(pso_config.get("n_iterations", 100))
        self.inertia_wt         = float(pso_config.get("inertia_weight", 1))
        self.c1                 = float(pso_config.get("c1", 2))
        self.c2                 = float(pso_config.get("c2", 2))

        # 验证环节 参数.
        val_config              = config["Validation"]
        self.validation_type    = val_config.get("type", "random-split")
        self.n_validations      = int(val_config.get("n_validations", 10))
        self.validation_size    = float(val_config.get("validation_size", 0.1))

        # SVR 参数.
        svr_config              = config["SVR"]
        self.kernel             = svr_config.get("kernel", "rbf")
        self.epsilon            = float(svr_config.get("epsilon", 0.1))
        self.C_min              = float(svr_config.get("C_min", 1))
        self.C_max              = float(svr_config.get("C_max", 100))
        self.C_step             = float(svr_config.get("C_step", 10))
        self.gamma_min          = float(svr_config.get("gamma_min", 0.01))
        self.gamma_max          = float(svr_config.get("gamma_max", 2))
        self.gamma_step         = float(svr_config.get("gamma_step", 0.1))

        # 数据路径
        data_config             = config["Data"]

        # 读取数据
        self.X, self.Y = load_data(type)

        # 初始化所有粒子（SVRs）
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
        # 所有的 SVRs (particles).
        self.svrs = []
        # 每个 svr 的局部最优 score
        self.p_best_score = []
        # 每个 svr 获得局部最优时的参数值
        self.p_best_params = []
        # 全局最优
        self.g_best_score = 0
        # 取得全局最优时的（C，gamma）参数值
        self.g_best_params = {}

        ctemp=[]
        gammatemp=[]
        # Initialize all the particle states and p_best values.
        for C in np.arange(self.C_min, self.C_max, self.C_step):
            for gamma in np.arange(self.gamma_min, self.gamma_max, self.gamma_step):
                ctemp.append(C)
                gammatemp.append(gamma)
                svr = SVR(kernel = self.kernel,
                           C = C,
                           epsilon = self.epsilon,
                           gamma = gamma)

                self.svrs.append(svr)
                self.p_best_score.append(0)
                self.p_best_params.append({"C":0, "gamma": 0})

        self.n_svrs = len(self.svrs)
        c_pair_plot.append(ctemp)
        gamma_pair_plot.append(gammatemp)


    def get_train_val_data(self, validation_step):
        """
        1. 将 data 划分为训练集与测试集
        2. 将 x_train 标准化为标准正态分布
        """

        if self.validation_type == "random-split":
            x_train, x_test, y_train, y_test = train_test_split(
                        self.X, self.Y, test_size = self.validation_size)
        elif self.validation_type == "k-fold":
            # n_validations 就是 k.
            k = self.n_validations
            n_samples = self.Y.shape[0]
            subset_size = np.ceil(n_samples/k)

            #选取第几个 subset 作为验证集
            idx_min = int((validation_step - 1) * subset_size)
            idx_max = validation_step * subset_size
            idx_max = int(min(idx_max, n_samples))

            # 验证集
            x_test = self.X[idx_min:idx_max]
            y_test = self.Y[idx_min:idx_max]
            # 测试集
            x_train = self.X[np.r_[0:idx_min, idx_max:n_samples]]
            y_train = self.Y[np.r_[0:idx_min, idx_max:n_samples]]

        # 数据标准化
        # 将 x_train 标准化为正态分布
        x_scaler = StandardScaler()
        # x_scaler = MinMaxScaler()
        # 将数据按期属性（按列进行）减去其均值，并处以其方差。对于每个属性 /每列来说所有数据都聚集在0附近，方差为1。
        x_train = x_scaler.fit_transform(x_train)
        # 根据已知的 mu 与 sigma 参数标准化 x_test
        x_test = x_scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    def run_optimizer(self):
        """
        优化器
        """
        start = time.time()

        for i in range(self.n_iterations):
            print("\nOptimizer step: ", i+1)

            # 训练 SVR，获得验证分数
            self.train_svrs()
            #根据验证分数，更新粒子(C and gamma)的状态
            self.update_particle_state()
            print ('training takes {:.2f} minutes'.format((time.time()-start)/60))
            print("Step best score: ", max(self.val_score))
            print("Global best score: ", self.g_best_score)


    def train_svrs(self):
        """
        函数用于训练所有的SVR，并获得验证分数
        """
        # 存储每个SVR的验证分数的列表的列表。
        # 每个子列表对应一个svr，子列表中的每个元素将是对应于每一个验证步骤的验证分数。

        val_score_list = [[] for i in range(self.n_svrs)]

        #多次训练
        for val_step in range(self.n_validations):
            # 获取训练数据以及得分
            print ("val_step + 1:",val_step + 1)
            x_train, x_test, y_train, y_test = self.get_train_val_data(
                        validation_step = val_step + 1)
            # 训练每一个svr
            for j in range(self.n_svrs):
                self.svrs[j].fit(x_train, y_train.ravel())

                # 获取评分.
                # y_pred = self.svrs[j].predict(x_test)
                # val_score = mean_squared_score(y_test, y_pred)
                val_score = self.svrs[j].score(x_test, y_test)
                val_score_list[j].append(val_score)

        # 将最终验证分数设置为每步验证分数的平均值。
        # 每一行都是一个svr的训练多次后得到的平均验证分数
        # print (len(val_score_list),len(val_score_list[0])) # 200,10
        self.val_score = np.mean(val_score_list, axis = 1)


    def update_particle_state(self):
        """
        更新每个粒子的局部和全局最佳值，然后更新每个粒子的参数
        """

        for i in range(self.n_svrs):
            # 更新 p_best
            if self.val_score[i] > self.p_best_score[i]:
                params = self.svrs[i].get_params()
                self.p_best_score[i] = self.val_score[i]
                self.p_best_params[i] = {'C': params['C'],
                            'gamma': params['gamma']}

                # 更新 g_best
                if self.p_best_score[i] > self.g_best_score:
                    self.g_best_score = self.p_best_score[i]
                    self.g_best_params = params


        ctemp=[]
        gammatemp=[]
        # 更新 SVRs 的参数值
        for i in range(self.n_svrs):
            params = self.svrs[i].get_params()

            r1 = np.random.random()
            r2 = np.random.random()

            # 更新 C 的值
            C = params["C"]
            C_new = self.inertia_wt * C \
                    + r1*self.c1*(self.p_best_params[i]["C"] - C) \
                    + r2*self.c2*(self.g_best_params["C"] - C)
            C_new = max(0.01, C_new)

            # Find new gamma value.
            gamma = params["gamma"]
            gamma_new = self.inertia_wt * gamma \
                    + r1*self.c1*(self.p_best_params[i]["gamma"] - gamma) \
                    + r2*self.c2*(self.g_best_params["gamma"] - gamma)
            gamma_new= max(0.001, gamma_new)
            ctemp.append(C_new)
            gammatemp.append(gamma_new)
            # 更新 SVR 的参数.
            self.svrs[i].set_params(**{'C': C_new, 'gamma': gamma_new})

        c_pair_plot.append(ctemp)
        gamma_pair_plot.append(gammatemp)




    def get_best_values(self):
        """
        返回获得的全局最优和获得全局最优的SVR的参数
        """
        return self.g_best_score, self.g_best_params

#-----------------------------------------------------------------------------#

#--------------------------------Final Trainer--------------------------------#

def train_svr(config_file, svr_params,type):
    """
    用最终获取的参数训练svr，并保存svr模型
    """
    # 加载数据.
    config = configparser.ConfigParser()
    config.read(config_file)
    data_config = config["Data"]
    X, Y = load_data(type)

    # 新建 pipeline（集合fit transform train）.
    # x_scaler为标准正态分布应用器
    x_scaler = StandardScaler()
    # x_scaler = MinMaxScaler()
    # 设定好参数的 svr
    svr = SVR(**svr_params)

    pipeline = Pipeline(steps = [('preprocess', x_scaler), ('SVR', svr)])
    pipeline.fit(X,Y)

    # Save the pipeline.
    if type=='speed':
        save_path = config["Model"]["save_path_s"]
        with open(save_path, 'wb') as save_file:
            pickle.dump(pipeline, save_file)

    if type=='flow':
        save_path = config["Model"]["save_path_f"]
        with open(save_path, 'wb') as save_file:
            pickle.dump(pipeline, save_file)

#-----------------------------------------------------------------------------#

#-------------------------------------MAIN------------------------------------#

if __name__ == "__main__":

    config_file = sys.argv[1]

    # Use PSO to find best hyper parameters for SVR.
    train_type = 'flow'
    print("Running PSO")
    pso_optimizer = PSO_SVR(type = train_type,config_file = config_file)

    start = time.time()
    pso_optimizer.run_optimizer()

    print ('The whole training takes {:.2f} minutes'.format((time.time()-start)/60))

    best_score, best_params = pso_optimizer.get_best_values()
    print("\nBest score: ", best_score)
    print("Params of best SVR : ", best_params)

    with open(train_type + '_best_params.json','w') as f:
        json.dump(best_params,f)


    # Use the best parameters to train an SVR using the whole data.
    print("\nTraining final model")
    train_svr(config_file, best_params,type=train_type)

    with open('c_' + train_type + '.json','w') as f:
        json.dump(c_pair_plot,f)
    with open('gamma_'+ train_type + '.json','w') as f:
        json.dump(gamma_pair_plot,f)


#-----------------------------------------------------------------------------#
