# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:55 2022

@author: dirty
"""

import pandas as pd
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import PM25Dataset
from model import PM25_LSTM_Regression
from torchmetrics import Accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    path = './'
    trainSet = pd.read_csv(path + 'EPA_OD_202209.csv')
    #testSet = pd.read_csv(path + 'test.csv')
    
    trainSet = trainSet[['SiteId', 'PM2.5','PublishTime']]
    # 空白資料補0
    trainSet = trainSet.fillna(value = 0) 
    # 資料轉置成橫向
    trainSet = trainSet.pivot(index='SiteId', columns='PublishTime', values='PM2.5')
    
    train,vaild = train_test_split(trainSet, test_size=0.1)

    
    train_dataset = PM25Dataset(train,'train')
    vaild_dataset = PM25Dataset(vaild,'train')
    
    
    
    # model parameter 
    hidden_size = 32
    num_layers = 2
    bidirectional = True
    intput_size = 6
    batch_size = 32
    
    # data loader
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    vaild_loader = DataLoader(vaild_dataset, batch_size = batch_size, shuffle = False)
    
    model = PM25_LSTM_Regression(hidden_size,num_layers,bidirectional,intput_size,batch_size).to(device)
    
    model.train()
    
    optimizer = optim.Adam(model.parameters(), 1e-3)
    #optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=0.01)
    epoch_size = 100
    epoch_pbar = trange(epoch_size, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loop(train_loader, model, optimizer,intput_size)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        vaild_loop(vaild_loader, model, optimizer,intput_size)
        pass

    
    ckpt_path = "./checkpoint/checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    
def train_loop(dataloader,model,optimizer,intput_size):
    
    loss_function = torch.nn.MSELoss()
    num_batches = len(dataloader)
    model.train()
    total_MSE = 0
    # model初始化
    model.zero_grad()
    
    for data in dataloader:
        
        total_lenth = data.shape[1]
        start_pos = 0
        end_pos = intput_size
        
        # 以前6項資料當作feature預測第7項
        for index in range(0,total_lenth - intput_size - 1,1):
            intput_data = data[:,start_pos + index : end_pos + index]
            predit_data = data[:,end_pos + index + 1]
        
            X = torch.from_numpy(np.array(intput_data)).float().to(device)
            y = torch.from_numpy(np.array(predit_data)).float().to(device)

            output = model(X)
            # 計算loss
            loss = loss_function(output,y)
    
            # 梯度清零
            optimizer.zero_grad()
            # 反向傳播
            loss.backward()
            # 更新參數
            optimizer.step()
        
        total_MSE += loss
    
    average_MSE = total_MSE / num_batches
    print(f"\n training MSE: {total_MSE} ")
    return average_MSE
    
def vaild_loop(dataloader,model,optimizer,intput_size):
    loss_function = torch.nn.MSELoss()
    num_batches = len(dataloader)
    model.eval()
    total_MSE = 0
    # model初始化
    model.zero_grad()
    
    for data in dataloader:
        
        total_lenth = data.shape[1]
        start_pos = 0
        end_pos = intput_size
        
        for index in range(0,total_lenth - intput_size - 1,1):
            intput_data = data[:,start_pos + index : end_pos + index]
            predit_data = data[:,end_pos + index + 1]
        
            X = torch.from_numpy(np.array(intput_data,dtype="float64")).float().to(device)
            y = torch.from_numpy(np.array(predit_data)).float().to(device)

            output = model(X)
            # 計算loss
            loss = loss_function(output,y)
     
        total_MSE += loss
    
    average_MSE = total_MSE / num_batches
    print(f"\n vaild MSE: {total_MSE} ")
    return average_MSE
    

if __name__ == "__main__":
    main()