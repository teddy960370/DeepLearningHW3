# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:02:11 2022

@author: dirty
"""

import pandas as pd
import numpy as np
from tqdm import trange
from tqdm import tqdm


import torch
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import PM25Dataset
from model import PM25_LSTM_Regression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    path = './'
    testData = pd.read_csv(path + 'Sample_Submission.csv')
    #testSet = pd.read_csv(path + 'test.csv')
    
    #testSet = testSet[['SiteId', 'PM2.5','PublishTime']]
    testData = testData.fillna(value = 0) # 空白資料補0
    testSet = testData.pivot(index='SiteId', columns='PublishTime', values='PM2.5')
    
    test_dataset = PM25Dataset(testSet,'test')
    
    # model parameter 
    hidden_size = 32
    num_layers = 2
    bidirectional = True
    intput_size = 6
    batch_size = 32
    
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    model = PM25_LSTM_Regression(hidden_size,num_layers,bidirectional,intput_size,batch_size).to(device)
    
    model.eval()
    
    ckpt_path = "./checkpoint/checkpoint.pt"
    ckpt = torch.load(ckpt_path)
    
    # load weights into model
    model.load_state_dict(ckpt)
    
    ans = test_model(test_loader,model,intput_size)
    
    predit_data = [0,0,0,0,0,0] + np.array(ans).tolist()
    
    
    testData['Predict_PM2.5'] = predit_data
    
    testData.to_csv('result.csv',index=False)
    
def test_model(dataloader, model,intput_size):
    loss_function = torch.nn.MSELoss()
    
    ans = np.array([])
    total_MSE = 0
    
    model.eval()
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            total_lenth = data.shape[1]
            start_pos = 0
            end_pos = intput_size
            
            for index in range(0,total_lenth - intput_size,1):
                intput_data = data[:,start_pos + index : end_pos + index]
                predit_data = data[:,end_pos + index]
                
                X = torch.from_numpy(np.array(intput_data,dtype="float64")).float().to(device)
                y = torch.from_numpy(np.array(predit_data)).float().to(device)
                
                output = model(X)
                
                loss = loss_function(output,y)
                total_MSE += loss
                
                output = output.round()      #四捨五入取整數
                ans = np.append(ans,output.to('cpu').detach().numpy())
                
                
    print(f"\n test MSE: {total_MSE} ")
    return ans

if __name__ == "__main__":
    main()