import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from time import time
import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import lightgbm as lgb

folder_path = "./system/engines/model/lgbtree"  
    #train,validation,tes
def train_test(target_feature,traceback,years,df,target):
        #Time series cross validation, use the data before traceback years.
        current_time = df.index[-1]
        time_threshold = current_time - pd.DateOffset(years = traceback)
        train_end = time_threshold
        validation_start = train_end
        validation_end = train_end + pd.DateOffset(years = years)

        #data&target
        data = df
        target = target

        #split and let the nan go
        missing_values = target[target_feature].isnull()
        missing_index = target.index[missing_values]
        target = target.drop(missing_index)
        data = data.drop(missing_index)

        feature=data.columns.tolist()
        X_train=data.loc[:train_end][feature].values
        y_train=target.loc[:train_end][target_feature].values
        X_validation=data.loc[validation_start:validation_end][feature].values
        y_validation=target.loc[validation_start:validation_end].values
        return X_train,y_train,X_validation,y_validation

def lgb_model_training(df,target_feature,target,traceback,years,num_leaves,learning_rate,num_boost):
    r2 = []
    mae = []
    mse = []
    print("start model training:")
    for i in traceback:
        X_train,y_train,X_validation,y_validation=train_test(target_feature,i,years,df,target)

        params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }

        #fit
        train_data = lgb.Dataset(X_train, label=y_train)
        lgb_reg = lgb.train(params, train_data, num_boost_round=num_boost)
        y_pred = lgb_reg.predict(X_validation)

       #evaluate
        y_test=y_validation.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
        r2.append(r2_score(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
    print(np.mean(mae))
    print(np.mean(r2))
    print(np.mean(mse))

    model_file ='lgb'+'-'+target_feature+'.txt'
    model_path = os.path.join(folder_path, model_file)
    lgb_reg.save_model(model_path)
    return np.mean(mae),np.mean(r2),np.mean(mse)

def lgb_model_testing(df,target_feature,target,traceback,years):
    #load data
    model_file ='lgb'+'-'+target_feature+'.txt'
    model_path = os.path.join(folder_path, model_file)
    model = lgb.Booster(model_file=model_path)

    traceback = [traceback[-1]]
    for i in traceback:
        _,_,X_validation,y_validation=train_test(target_feature,i,years,df,target)
        y_pred=model.predict(X_validation)

        #calculate metrics of validation set
        y_test=y_validation.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
    print(r2)
    print(mae)
    print(mse)
    X_test = df.loc[df.index[-1]].values
    y_test = model.predict(X_test)
    return mae,r2,mse,y_test

def plot_figure(df,target_feature,target,Years):
    model_file ='lgb'+'-'+target_feature+'.txt'
    model_path = os.path.join(folder_path, model_file)
    model = lgb.Booster(model_file=model_path)
    y_preds = []
    for i in range(len(df)):
        x_pred = df.iloc[i].values.reshape(1,14)
        y_pred = model.predict(x_pred)
        y_preds.append(y_pred)
    print(len(y_preds))
    print(target.shape)

    plt.plot(Years[1:],y_preds[:-1],label = 'y_pred')
    plt.plot(Years[1:],target[1:],label = 'y_real')
    plt.xlabel('year')
    plt.ylabel('target value')
    plt.title("Prediction of Feature")
    plt.legend()
    plt.show()
    return plt

def data_process(data,target_feature,max_lack):
    #load the data
    df = data
    df = df.dropna(subset=[target_feature])

    #reserve the lines with more than length of columns minus 2
    columns_length = df.shape[1]
    df.dropna(thresh=columns_length-max_lack, inplace=True)
    df.head()
    df['year'] = pd.to_datetime(df['year'],format='%Y')
    df = df.sort_values(by=['year'])
    df = df.set_index(['year'])
    df.drop('id', axis=1, inplace=True)
    df.head()
    # transform the country name to the one-hot
    int_encoded = pd.factorize(df['country'])
    df['country'] = int_encoded[0]

    #standarization, firstly extract the target
    target_temp = df[['country',target_feature]]
    target_temp = target_temp.reset_index()
    target_temp[target_feature] = target_temp.groupby('country')[target_feature].shift(-1)
    target_temp = target_temp.set_index(['year'])
    target = target_temp.drop('country', axis=1)
    return df,target,int_encoded[1]

def lgbtree(target_feature,df,target, max_lack = 2 , traceback = [20,15,10,5] , years = 5 ,num_leaves=50,learning_rate=0.1,num_boost=1000,train = 1):
    #create a scaler
    temp = df.iloc[:,1:]
    scaler = StandardScaler()
    temp = scaler.fit_transform(temp)
    df.iloc[:,1:] = temp
    df.head()
    if train == 1:
        mae,r2,mse = lgb_model_training(df,target_feature,target,traceback,years,num_leaves,learning_rate,num_boost)
        y_test = None
        #target_feature and target must be the same
    else:
        mae,r2,mse,y_test = lgb_model_testing(df,target_feature,target,traceback,years)
    return mae,r2,mse,y_test
 
def run(data,country,target_feature = 'gas_product',train = 1):
    traceback = [10,8,6,4,2]
    years = 2
    max_lack = 2
    df,target,dic = data_process(data,target_feature,max_lack)
    mae,r2,mse,y_test = lgbtree(target_feature,df,target,train = train,max_lack = max_lack, traceback = traceback , years = years  ,num_leaves=80,learning_rate=0.1,num_boost=1000)
    idx = np.where(dic == country)[0][0]
    Years = df[df['country'] == idx]
    df.reset_index()
    Years = Years.index
    figure = plot_figure(df[df['country'] == idx],target_feature,target[df['country'] == idx],Years)
    return r2,mse,mae,y_test[idx],figure

