#!/usr/bin/env python
# coding: utf-8

# In[56]:


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
import joblib
import pickle

class EMBModel(nn.Module):
    def __init__(self, num_ids,feature_types = ['dense','sparse'],emb_size = 4,proba_dropout = 0):
        # 默认每个特征都是1列，这个是最粗略的版本
        super().__init__()
        self.embs = nn.ModuleList()
        self.feature_types = feature_types
        self.sparse_num = 0 #稀疏特征的数量
        self.dense_num = 0 #稠密特征的数量
        self.dropout = nn.Dropout(proba_dropout)
        for feature_type in feature_types:
            if feature_type == 'dense':
                self.dense_num += 1

            elif feature_type == 'sparse':
                self.embs.append(nn.Embedding(num_ids, emb_size))
                self.sparse_num += 1
    def forward(self,x):
        '''
        x输入是三维，(batch_size,num_steps,num_feature)
        '''
        output = []
        emb_index = 0
        for index in range(len(self.feature_types)):
            feature_type = self.feature_types[index]
            if feature_type == 'dense':
                res = x[:,:,index:index+1]
                output.append(res)
            elif feature_type == 'sparse':
                res = self.embs[emb_index](x[:,:,index].int())
                res = self.dropout(res)
                output.append(res)
        output = torch.cat(output,dim=2)
        return output

class RNNModel(nn.Module):
    #循环神经网络模型
    def __init__(self,input_size,output_size,num_hiddens,embModel = None,RevINModel = None,num_directions = 1, num_layers = 1,drop_proba = 0):
        super(RNNModel,self).__init__()
        #self.rnn = nn.GRU(input_size, num_hiddens,num_layers,batch_first=True)
        self.rnn = nn.LSTM(input_size, num_hiddens,num_layers,batch_first=True)
        #for p in self.rnn.parameters():
            #nn.init.normal_(p, mean=0.0, std=0.001)
        self.input_size = input_size
        self.dropout = nn.Dropout(drop_proba)
        self.output_size = output_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 定义词嵌入层
        if embModel is None:
            self.use_emb = False
        else:
            self.use_emb = True
            self.embModel = embModel
        # 定义RevIn层
        if RevINModel is None:
            self.use_rev = False
        else:
            self.use_rev = True
            self.RevINModel = RevINModel
        self.num_directions = num_directions
        self.output = nn.Linear(self.num_hiddens,self.output_size)#设计输出层

    def forward(self,X,state):
        if self.use_emb: X = self.embModel(X)
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
    '''
    if len(x) < num_steps:
        print('输入序列长度太短，无法预测')
        return
    else:
        #拟合
        if state is None: state = net.begin_state(batch_size = 1)
        len_x = len(x)
        x = x.reshape(1,-1,input_size)
        target = target.reshape(-1,output_size)
        yhat = []
        state = net.begin_state(batch_size = 1)
        for i in range(num_steps,len_x):
            output,state = net(torch.tensor(x[:,i-num_steps:i,:],dtype = torch.float32),state)
            #更新预测值
            yhat.append(output[-1,:].detach().numpy()) #取决于num_delays
    '''
    if state is None: state = net.begin_state(batch_size = 1)
    yhat,_ = net(torch.tensor(np.asarray(x)[None,:],dtype=torch.float32),state)

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
            #print('{} of the dataset on {} is {}'.format(metric, cur_names[i],res[i][j]))
    return res

def plot_result(year,yhat,x,ind,countries = ['Unknown'],input_size = 1):
    '''
    Input: x,yhat of size (num_curves, len_sequences)
    Output:
    '''
    #ind = int(len(x)*train_test_rate)
    # 画拟合曲线
    num_curves = len(yhat)
    #fig,axes = plt.subplots(num_curves, 1,figsize = (20,15))
    fig,axes = plt.subplots(num_curves, 1)
    if num_curves > 1:
        for i in range(num_curves):
            axes[i].plot(year,x[:,i],label='Real Data')
            axes[i].plot(year,yhat[i],label='Fitted Data') #画拟合曲线
            axes[i].set_xlabel('year')
            axes[i].set_ylabel('target value')
            axes[i].set_title('Prediction of Feature')
            axes[i].legend()
            axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
            axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
            # 在x轴顶部对应比例位置添加文本
            text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
            #axes[i].text((year[0]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
            #axes[i].text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)
    else:
        axes.plot(year,x[0],label='Real data')
        axes.plot(year,yhat[0],label='Fitted data') #画拟合曲线
        axes.set_xlabel('year')
        axes.set_ylabel('target value')
        axes.set_title('Prediction of Feature')
        axes.legend()
        axes.axvline(x=year[ind], color='r', linestyle='--') #画辅助线
        # 在x轴顶部对应比例位置添加文本
        text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
        #axes.text((year[0]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
        #axes.text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)

    return fig


from sklearn.preprocessing import MinMaxScaler
# 进行归一化
# 进行归一化
def normalized(x,scaler = None):
  #x和normalized_x都是二维，(length,1)；
  #创建 MinMaxScaler 对象
    if scaler is None:
        scaler = MinMaxScaler()
        # 对数据集进行归一化
        normalized_x = scaler.fit_transform(x)
        return normalized_x,scaler
    else:
        normalized_x = scaler.transform(x)
        return normalized_x,scaler

# 进行embedding
def data_list(data,id_encoder):
    data['enc_id'] = id_encoder.transform(data['id'])
    data.drop('id',axis=1,inplace = True)
    return data,id_encoder


def load_lstm(data, start, tail,target_name=None):
    # 读取数据
    if target_name is None:
        target_name = 'oil_price'
    
    # 读取模型
    net = torch.load('./system/engines/model/Lstm_cheng/LSTM_240427/LSTM_oil_price2.pth')
    
    #读取数据参数
    with open('./system/engines/model/Lstm_cheng/LSTM_240427/id_set.pkl', 'rb') as f:
        ids = pickle.load(f) #读取国家ids
    
    scaler = joblib.load('./system/engines/model/Lstm_cheng/LSTM_240427/min_max_scaler.pkl')#读取label_encoder
    id_encoder = joblib.load('./system/engines/model/Lstm_cheng/LSTM_240427/label_encoder.pkl')#读取min_max_encoder
    
    #初始化数据
    train_test_rate = 0.85
    train_end_ind = int((tail-start)*train_test_rate)
    
    #处理数据
    year,data = data.iloc[:,0],data.iloc[:,1:]
    year = np.array(year.drop_duplicates())
    data = data[['id',target_name]]
    tmp,scaler = normalized(np.array(data[target_name]).reshape(-1,1),scaler=scaler) #正则化
    data[target_name] = pd.Series(tmp.reshape(-1))
    data,_ = data_list(data,id_encoder)
    
    #预测
    fig_dict, train_metric_dict, test_metric_dict = {},{},{}
    for id1 in list(ids):
        id2 = id_encoder.transform([id1])[0]
        #print(id2)
        yhat = predict(net,np.array(data[data['enc_id']==id2]),target = np.array(data[target_name][data['enc_id']==id2]),state = None)
        #画出结果
        x = np.array(data[target_name][data['enc_id']==id2]).reshape(-1,1)
        fig_dict[id1] = plot_result(year,yhat.T,x.T,train_end_ind)
        # 输出结果
        train_metric_dict[id1] = print_metric(x.T[:,:train_end_ind],yhat.T[:,:train_end_ind])[0]
        test_metric_dict[id1] = print_metric(x.T[:,train_end_ind:],yhat.T[:,train_end_ind:])[0]
    return fig_dict, train_metric_dict, test_metric_dict

if __name__ == '__main__':
    start,tail = 1980,2013
    data = pd.read_csv('./system/engines/Lstm_cheng/LSTM_240427/data_Cheng.csv')
    fig_dict, train_metric_dict, test_metric_dict = load_lstm(data, start, tail,target_name=None)


