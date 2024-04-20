# 输入：国家，特征，年份，
# 输出：评价指标和图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

class RNNModel(nn.Module):
    #循环神经网络模型
    def __init__(self,input_size,num_hiddens,num_directions = 1, num_layers =1, drop_proba=0):
        super(RNNModel,self).__init__()
        #self.rnn = nn.GRU(input_size, num_hiddens,num_layers,batch_first=True)
        self.rnn = nn.LSTM(input_size, num_hiddens,num_layers,batch_first=True)
        #for p in self.rnn.parameters():
            #nn.init.normal_(p, mean=0.0, std=0.001)
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = nn.Dropout(drop_proba)
        self.output = nn.Linear(self.num_hiddens,self.input_size)#设计输出层

    def forward(self,X,state):
        Y,state = self.rnn(X,state)
        Y = self.dropout(Y)
        Y = Y.reshape(-1,self.num_hiddens)
        output = self.output(Y)
        return output,state

    def begin_state(self,batch_size = 1):
        return (torch.zeros(size = (self.num_layers*self.num_directions,batch_size,self.num_hiddens)),
                torch.zeros(size = (self.num_layers*self.num_directions,batch_size,self.num_hiddens)))


def predict(net,x,num_preds=0,input_size = 1,output_size=1,num_delays = 1,state = None,target = None):
    '''
    默认Output_size是1; yhat的输出是(output_size, len_sequences)
    '''
    if state is None: state = net.begin_state(batch_size = 1)
    print(x.shape)
    yhat,_ = net(torch.tensor(np.array(x).reshape(1,-1,1),dtype=torch.float32),state)

    return yhat.detach().numpy()

def calculate_metric(y_true, y_pred, metric):
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError('Unsupported metric. Please choose from "r2" or "mse".')

def print_metric(y_true, y_pred,metrics = ['r2','mse','mae'],cur_names = None):
    num_curves = len(y_true)
    if cur_names is None:
        cur_names = ['' for _ in range(num_curves)]
    res = np.zeros((num_curves,len(metrics)))
    for i in range(num_curves):
        for j,metric in enumerate(metrics):
            res[i][j] = calculate_metric(y_true[i],y_pred[i],metric)
            print('{} of the dataset on {} is {}'.format(metric, cur_names[i],res[i][j]))
    return res

def plot_result(yhat,x,ind,countries = ['Unknown'],input_size = 1):
    '''
    Input: x,yhat of size (num_curves, len_sequences)
    Output:
    '''
    #ind = int(len(x)*train_test_rate)
    # 画拟合曲线
    num_curves = len(yhat)
    fig,axes = plt.subplots(num_curves, 1,figsize = (20,15))
    if num_curves > 1:
        for i in range(num_curves):
            axes[i].plot(year,x[:,i],label='Real data')
            axes[i].plot(year,yhat[i],label='Fitted data') #画拟合曲线
            axes[i].set_xlabel('time')
            axes[i].set_ylabel('value')
            axes[i].legend()
            axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
            axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
            # 在x轴顶部对应比例位置添加文本
            text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
            axes[i].text((year[num_steps]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
            axes[i].text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)
    else:
        axes.plot(year,x[0],label='Real data')
        axes.plot(year,yhat[0],label='Fitted data') #画拟合曲线
        axes.set_xlabel('time')
        axes.set_ylabel('value')
        axes.legend()
        axes.axvline(x=year[ind], color='r', linestyle='--') #画辅助线
        # 在x轴顶部对应比例位置添加文本
        text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
        axes.text((year[num_steps]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
        axes.text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)

    return fig


from sklearn.preprocessing import MinMaxScaler
# 进行归一化
def normalized(x):
    #创建 MinMaxScaler 对象
    scaler = MinMaxScaler()
    # 对数据集进行归一化
    normalized_x = scaler.fit_transform(x)
    return normalized_x,scaler


def load_lstm(data,country, feature_name, start, tail):

    # 获取数据
    #net = torch.load("./system/engines/model/Lstm/"+country+'_'+feature_name+'_'+'model3.pth')
    net = torch.load("./system/engines/model/Lstm/"+'Lstm-'+country+'-'+feature_name+'.pth')
    #处理数据
    input_size=1
    x = data[(start<=data['year']) & (data['year']<=tail)&(data['country'] == country)]
    year,x = np.array(x['year']),np.array(x[[feature_name]]).reshape(-1,1)
    train_test_rate = 0.8
    ind = int(len(year)*train_test_rate)
    normalized_x,scalar=normalized(x)
    #预测
    yhat = predict(net,normalized_x,num_preds=0,input_size = input_size,num_delays = 1,state = None)
    #画出结果
    yhat, x = yhat.reshape(1,-1),normalized_x.reshape(1,-1)
    fig = plot_result(yhat,x,ind)
    # 输出结果
    training_metrics = print_metric(x[:,:ind],yhat[:,:ind])
    test_metrics = print_metric(x[:,ind:],yhat[:,ind:])
    return fig, training_metrics[0], test_metrics[0]