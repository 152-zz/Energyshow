# 输入：国家，特征，年份，
# 输出：评价指标和图(240501)
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

def predict(net,x,num_preds=0,input_size = 1,output_size=1,num_delays = 1,state = None,target = None,seq_len=None):
    '''
    默认Output_size是1; yhat的输出是(output_size, len_sequences)
    '''
    if state is None: state = net.begin_state(batch_size = 1)
    if seq_len is None:
        yhat,_ = net(torch.tensor(np.asarray(x)[None,:],dtype=torch.float32),state)
        return yhat.detach().numpy()
    else:
        num_seq = x.shape[0] // seq_len
        yhat_all = np.zeros((x.shape[0],1))
        for i in range(num_seq):
            state = net.begin_state(batch_size = 1)
            yhat,_ = net(torch.tensor(np.asarray(x[i*seq_len:(i+1)*seq_len,:])[None,:],dtype=torch.float32),state)
            yhat_all[i*seq_len:(i+1)*seq_len,:] = yhat.detach().numpy()
        return yhat_all

def calculate_metric(y_true, y_pred, metric):
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError('Unsupported metric. Please choose from "r2" or "mse".')

def print_metric(y_true, y_pred,metrics = ['r2','mse','mae'],cur_names = None,is_print=True):
    num_curves = len(y_true)
    if cur_names is None:
        cur_names = ['' for _ in range(num_curves)]
    res = np.zeros((num_curves,len(metrics)))
    for i in range(num_curves):
        for j,metric in enumerate(metrics):
            res[i][j] = calculate_metric(y_true[i],y_pred[i],metric)
            if is_print:print('{} of the dataset on {} is {}'.format(metric, cur_names[i],res[i][j]))
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
            #axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
            #axes[i].axvline(x=year[ind], color='r', linestyle='--') #画辅助线
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
        axes.text((year[0]+year[ind])//2, text_y_position-1.315, 'Train set', ha='center',color="red",fontsize=14)
        axes.text((year[-1]+year[ind])//2+0.05, text_y_position-1.32, 'Test set', ha='center',color="red",fontsize=14)
    return fig


from sklearn.preprocessing import MinMaxScaler
# 进行归一化
def normalized(data,target_name,selected_index,MinMaxScalerLst,seq_len):
    '''
    data是只包含enc_id和一个目标特征列；直接从本质上修改了输入的data
    '''
    for i in range(len(selected_index)):
        # 新建一个
        scaler = MinMaxScalerLst[selected_index[i]]
        data[target_name][i*seq_len:(i+1)*seq_len] = pd.Series(scaler.transform(np.array(data[target_name][i*seq_len:(i+1)*seq_len]).reshape(-1,1)).reshape(-1))

# 进行反归一化
def denormalized(data,target_name,selected_index,MinMaxScalerLst,seq_len):
    '''
    data是只包含enc_id和一个目标特征列；直接从本质上修改了输入的data
    '''
    for i in range(len(selected_index)):
        # 新建一个
        scaler = MinMaxScalerLst[selected_index[i]]
        data[target_name][i*seq_len:(i+1)*seq_len] = pd.Series(scaler.inverse_transform(np.array(data[target_name][i*seq_len:(i+1)*seq_len]).reshape(-1,1)).reshape(-1))

# 进行embedding
def data_list(data,id_encoder):
    data['enc_id'] = id_encoder.transform(data['id'])
    data.drop('id',axis=1,inplace = True)
    return data,id_encoder


def load_lstm(data, start, tail,target_name=None):
    # 初始化参数
    train_test_rate = 0.85
    train_end_ind = int((tail-start)*train_test_rate)
    seq_len = tail - start + 1
    # 导入模型附属文件
    path = "./system/engines/model/Lstm_cheng/"
    with open(path+target_name+'/id_set_'+target_name+'.pkl', 'rb') as f:
        selected_index = pickle.load(f) #读取国家ids

    with open(path+target_name+'/MinMaxScalerLst_'+target_name+'.pkl', 'rb') as f:
        MinMaxScalerLst = pickle.load(f) #读取国家ids

    id_encoder = joblib.load(path+'oil_price/label_encoder_oil_price.pkl')
    net = torch.load(path+target_name+'/LSTM_'+target_name+'.pth')
    
    # 初始化数据
    year = np.array(data['year'].drop_duplicates())
    data = data[['id']+[target_name]]
    
    selected_ids = id_encoder.inverse_transform(np.array(selected_index))
    data = data[data['id'].isin(selected_ids)].reset_index(drop=True) # 只保留selected ids的部分的数据
    normalized(data,target_name,selected_index,MinMaxScalerLst,seq_len)
    data,_ = data_list(data,id_encoder) #返回预测值
    #预测
    fig_dict, train_metric_dict, test_metric_dict = {},{},{}
    for id1 in list(selected_ids):
        id2 = id_encoder.transform([id1])[0]
        yhat = predict(net,np.array(data[data['enc_id']==id2]),target = np.array(data[target_name][data['enc_id']==id2]),state = None)
        #画出结果
        x = np.array(data[target_name][data['enc_id']==id2]).reshape(-1,1)
        fig_dict[id1] = plot_result(year,yhat.T,x.T,train_end_ind)
        # 输出结果
        train_metric_dict[id1] = print_metric(x.T[:,:train_end_ind],yhat.T[:,:train_end_ind],is_print=False)[0]
        test_metric_dict[id1] = print_metric(x.T[:,train_end_ind:],yhat.T[:,train_end_ind:],is_print=False)[0]
    return fig_dict, train_metric_dict, test_metric_dict

if __name__ == '__main__':
    # 读取属性数据
    target_name = 'gas_product'
    start,tail = 1980,2013
    data = pd.read_csv('./system/engines/model/Lstm_cheng/data1.csv')
    fig_dict, train_metric_dict, test_metric_dict = load_lstm(data, start, tail,target_name=target_name)
