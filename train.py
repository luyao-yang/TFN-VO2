from TFN.tft_model import  QuantileLoss
from TFN.tft_model import TemporalFusionTransformer
from TFN.tft_base_model import BaseTemporalFusionTransformer
from TFT_dataset import TFT_Dataset
from TFT_dataset_test import TFT_Dataset_test
import time
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim



import pdb
import os
import torch.nn as nn
 
class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        num = torch.abs(y_true - y_pred)
        denom = torch.abs(y_true) + torch.abs(y_pred) + self.eps
        smape = torch.mean(200. * num / denom)
        return smape
    
    
def read_data(params):
    model_base = params['model_base']
    data_dir="2s_data"
    # data_dir="3s_data"
    train_df = pd.read_csv(os.path.join(data_dir,"./train.csv"))
    train_df = train_df.drop(['BMI','VE'], axis=1)
    test_df = pd.read_csv(os.path.join(data_dir,"./test.csv"))
    test_df = test_df.drop(['BMI','VE'], axis=1)
    all_df = pd.read_csv(os.path.join(data_dir,"./Trans_CPET.csv"))
    all_df = all_df.drop(['BMI','VE'], axis=1)

    
    input_columns = ["Sex","Age", "Height", "Weight","Status","WorkLoad","HR","HRR","VT","BF"]
    
    if model_base:
        train_df = train_df.drop(columns=["WorkLoad","Sex","Age","Height","Weight","Status"])
        test_df = test_df.drop(columns=["WorkLoad","Sex","Age","Height","Weight","Status"])
        input_columns = ["HR","HRR","VT","BF"]

    target_column = ["VO2"]
    id_column = "ID"
    time_column = "Time"
    col_to_idx = {col: idx for idx, col in enumerate(input_columns)}
    
    ### ADD
    # train_df = train_df.drop(['ID'], axis=1)
    # test_df = test_df.drop(['ID'], axis=1)
    # all_df = all_df.drop(['ID'], axis=1)


    # encoder_steps = params['encoder_steps']
    # pdb.set_trace()
    decoder_steps = params['decoder_steps']
    training_data = TFT_Dataset(train_df, id_column, time_column, target_column, input_columns, decoder_steps)
    testing_data = TFT_Dataset_test(test_df, id_column, time_column, target_column, input_columns, decoder_steps)
    
    # pdb.set_trace()
    all_data = TFT_Dataset_test(all_df, id_column, time_column, target_column, input_columns, decoder_steps)

    return training_data, testing_data, all_data, col_to_idx


def train(params,train_dataloader,model,device):
    
    criterion1 = QuantileLoss(params["quantiles"])
    criterion2 = nn.MSELoss()
    criterion3 = nn.L1Loss()
    
    # criterion = torch.nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    print_every_k = 100
    losses = []

    switch_epoch = params['switch_epoch']
    for epoch in range(params["epochs"]):
        t0 = time.time()
        print(f"===== Epoch {epoch+1} =========")
        epoch_loss = 0.0
        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):

            # pdb.set_trace()
            labels = batch['outputs'][:,:,0].flatten().float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # pdb.set_trace()
            outputs, attention_weights = model(batch)

            if epoch < switch_epoch:
                loss = criterion1(outputs, labels)
            else:
                # loss = criterion2(outputs[:,1], labels)
                loss2 = criterion2(outputs[:,1], labels)
                loss3 = criterion3(outputs[:,1],labels)
                # loss = 0.5*loss2 + 0.5*loss3
                loss = loss2*0.2+loss3*0.8

            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i+1) % print_every_k == 0:
                # pdb.set_trace()
                print(f"Mini-batch {i+1} average loss: {round(running_loss / print_every_k, 5)}")
                running_loss = 0.0
        
        t1 = time.time()
        # pdb.set_trace()
        print(f"\nEpoch trained for {round(t1-t0, 2)} seconds")
        print("\nEpoch loss:", round(epoch_loss / (i+1), 5), "\n")
        losses.append(epoch_loss / (i+1))


        # If epoch is a multiple of 10, save the model
        tfcn_dir_path = "./models/tfcn_models"
        # dir_path = os.path.join("./models",module_name,"tft_models")
        if not os.path.exists(tfcn_dir_path):
            os.makedirs(tfcn_dir_path)
        
        # dir_path = os.path.join("./models",module_name,"tft_base_models")
        tfcn_dir_base_path = "./models/tfcn_base_models"
        if not os.path.exists(tfcn_dir_base_path):
            os.makedirs(tfcn_dir_base_path)
            
        tflstm_dir_path = "./models/tflstm_models"
        # dir_path = os.path.join("./models",module_name,"tft_models")
        if not os.path.exists(tflstm_dir_path):
            os.makedirs(tflstm_dir_path)
        
        # dir_path = os.path.join("./models",module_name,"tft_base_models")
        tflstm_dir_base_path = "./models/tflstm_base_models"
        if not os.path.exists(tflstm_dir_base_path):
            os.makedirs(tflstm_dir_base_path)
        
        if params['model_base'] and params['block']=='TCN':
            model_save_path = tfcn_dir_base_path
        elif params['model_base'] and params['block']=='LSTM':
            model_save_path = tflstm_dir_base_path
        elif not params['model_base'] and params['block']=='TCN':
            model_save_path = tfcn_dir_path
        elif not params['model_base'] and params['block']=='LSTM':
            model_save_path = tflstm_dir_path

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(model_save_path, f"checkpoint_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },save_path)


def main():
    with open('config.json','r') as f:
        params = json.load(f)


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE="cpu"

    training_data, testing_data, all_data, col_to_idx = read_data(params)

    params['col_to_idx'] = col_to_idx
    batch_size = params['batch_size']
    model_base = params['model_base']

    # pdb.set_trace()
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=2, shuffle=False)


    if model_base:
        model = BaseTemporalFusionTransformer(params,DEVICE)
    else:
        model =TemporalFusionTransformer(params, DEVICE)

    model.to(DEVICE)
    train(params, train_dataloader, model, DEVICE)

if __name__ == '__main__':
        main()
        
