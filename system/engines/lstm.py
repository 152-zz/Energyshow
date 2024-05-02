import pandas as pd
import numpy as np
#from xgboost import XGBRegressor as XGBR
#import xgboost
from time import time
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class myLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super(myLSTM, self).__init__()
        #self.ln = nn.LayerNorm([20,input_size])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, x):
        #x = self.ln(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])  # 只取最后一个时间步的输出
        return out

def datapre(n_timestamp=3,target = 'oil_price',time_threshold = 2011):
    #load the data
    data = pd.read_csv('./system/dataset/data.csv')
    columns_length = data.shape[1]
    data.drop('id', axis=1, inplace=True)
    # transform the country name to the int
    int_encoded = pd.factorize(data['country'])   
    data['country'] = int_encoded[0]
    country_to_number = dict(zip(int_encoded[1], int_encoded[0]))
    
    df = data[data['year'] <= time_threshold]
    df = df[df['year'] >= 1970]
    #data processing:
    def gettingid(data,country):
        data_id = data[data['country'] == country]
        return data_id
    X = []
    y = []
    def data_split(sequence, n_timestamp,X,y):
        ix = sequence.values
        iy =  sequence.loc[:,target].values
        for i in range(len(sequence)):
            end_ix = i + n_timestamp
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = ix[i:end_ix], [iy[end_ix]]
            X.append(seq_x)
            y.append(seq_y)
        return X,y
    for i in set(df['country'].values):
        d = gettingid(df,i)
        X,y = data_split(d, n_timestamp,X,y)
    X_train, y_train = np.array(X),np.array(y)
    X_train = torch.from_numpy(X_train).float()
    #X_total = torch.where(torch.isnan(X_total), torch.tensor(0), X_total)
    y_train = torch.from_numpy(y_train).float()
    print(X_train.shape)
    print(y_train.shape)
    
    df = data[data['year'] > time_threshold-n_timestamp]
    #data processing:
    X = []
    y = []
    for i in set(df['country'].values):
        d = gettingid(df,i)
        X,y = data_split(d, n_timestamp,X,y)
    X_test, y_test = np.array(X),np.array(y)
    X_test = torch.from_numpy(X_test).float()
    #X_total = torch.where(torch.isnan(X_total), torch.tensor(0), X_total)
    y_test = torch.from_numpy(y_test).float()    
    
    def rn(tensorx,tensory):
        mask = torch.isnan(tensorx).any(dim=(1, 2))
        tensorx = tensorx[~mask]
        tensory = tensory[~mask]
        mask = torch.isnan(tensory).any(dim=(1))
        tensorx = tensorx[~mask]
        tensory = tensory[~mask]
        return tensorx,tensory
    X_train,y_train = rn(X_train,y_train)
    X_test,y_test = rn(X_test,y_test)
    
    return X_train,y_train,X_test,y_test,country_to_number

def train_model(target='gas_exports',CTY = 'United States',input_size = 15,hidden_size = 120,num_layers =2,output_size = 1,learning_rate = 0.1,num_epochs = 3000,batch_size = 70,train=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train,y_train,X_va,y_va,country_to_number = datapre(n_timestamp=18,target = target,time_threshold = 2011)
    model = myLSTM(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if train == 1:
        for epoch in range(num_epochs):
            for i, (batch_X, batch_y) in enumerate(dataloader):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                # 打印损失
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    print('train',loss)
    
                    X_va = X_va.to(device)
                    y_va = y_va.to(device)
                    outputs = model(X_va)
                    #mse
                    loss = criterion(outputs, y_va)
                    #mae
                    absolute_diff = torch.abs(outputs - y_va)
                    mae = torch.mean(absolute_diff)
                    #r2
                    mean_targets = torch.mean(y_va)
                    total_sum_squares = torch.sum((y_va - mean_targets) ** 2)
                    residual_sum_squares = torch.sum((outputs - y_va) ** 2)
                    r2 = 1 - (residual_sum_squares / total_sum_squares)
          
                    print(f'Epoch [{epoch+1}/{num_epochs}], mseLoss: {loss.item():.4f},maeLoss:{mae:.4f},r2:{r2:.4f}')
                    if r2>=0.8:
                        break
        torch.save(model.state_dict(), './system/engines/model/Lstm/model'+target+'.pth')
    else:
        data = pd.read_csv('./system/dataset/data.csv')
        columns_length = data.shape[1]
        data.drop('id', axis=1, inplace=True)
        # transform the country name to the int
        int_encoded = pd.factorize(data['country'])   
        data['country'] = int_encoded[0]
        model.load_state_dict(torch.load('./system/engines/model/Lstm/model'+target+'.pth',map_location=torch.device('cpu')))
        number = country_to_number.get(CTY)

        fig_end = 2014
        used_back_years = 7
        fig_start = 1997-used_back_years
        df = data[data['year'] <= fig_end]
        df = df[df['year'] >= fig_start]
        #data processing:
        def gettingid(data,country):
            data_id = data[data['country'] == country]
            return data_id
        X = []
        def data_split(sequence, X,n_timestamp=18):
            ix = sequence.values
            iy = sequence.loc[:,target].values[-used_back_years:]
            print(sequence.columns)
            for i in range(len(sequence)):
                end_ix = i + n_timestamp
                if end_ix > len(sequence):
                    break
                seq_x= ix[i:end_ix]
                X.append(seq_x)
            return X,iy
        d = gettingid(df,number)
        X_test,y_test= data_split(d, X,n_timestamp=18)
        X_test = np.array(X)
        X_test = torch.from_numpy(X_test).float()
        X_test = X_test.to(device)
        
    #remove nan
        mask = torch.isnan(X_test)
        X_test = torch.where(mask, torch.zeros_like(X_test), X_test)
        y_test = np.array(y_test)
        y_test = torch.from_numpy(y_test).float()
        y_test = y_test.to(device)
        
    #remove nan
        mask = torch.isnan(y_test)
        y_test = torch.where(mask, torch.zeros_like(y_test), y_test)
        print(X_test)
        y_pred = model(X_test)
        
    #validation 
        X_va = X_va.to(device)
        y_va = y_va.to(device)
        outputs = model(X_va)
        #mse
        loss = criterion(outputs, y_va)
        #mae
        absolute_diff = torch.abs(outputs - y_va)
        mae = torch.mean(absolute_diff)
        #r2
        mean_targets = torch.mean(y_va)
        total_sum_squares = torch.sum((y_va - mean_targets) ** 2)
        residual_sum_squares = torch.sum((outputs - y_va) ** 2)
        r2 = 1 - (residual_sum_squares / total_sum_squares)
    #y_test is the real data and y_pred is predicted by X_test 
        print(y_test.shape)
        print(y_pred.shape)

    #draw figure
        y_pred = y_pred.detach().numpy().reshape(-1)
        y_test = y_test.detach().numpy()
        times = range(fig_end-used_back_years+1,fig_end+2)

        fig = plt.figure()
        plt.plot(times, y_pred)
        plt.plot(times[:-1], y_test)
        plt.legend(['prediction','real value'])
        plt.title(target+' of '+CTY)
        plt.xlabel('year')
        plt.ylabel(target+'value')
        
        # 显示图像
        plt.show()
                
    return float(r2),float(loss),float(mae),y_pred[-1],fig