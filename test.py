from data_pre import inverse_transform_categoricals
from data_pre import inverse_transform_reals
from data_pre import fit_preprocessing
from TFN.tft_model import TemporalFusionTransformer
from TFT_dataset_test import TFT_Dataset_test
from torch.utils.data import DataLoader
from TFN.tft_base_model import BaseTemporalFusionTransformer
from scipy import stats

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

from sklearn.metrics import r2_score
import matplotlib.cm as cm
import os
import json
import pdb
import torch
import time

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def bland_altman_plot(y_true_list, y_pred_list, ax, group_id):
    """Create a Bland-Altman plot for a specific group with unique colors for each sample."""
    
    # Flatten the lists of Series into a single array
    y_true = np.concatenate([y_true.values for y_true in y_true_list])
    y_pred = np.concatenate([y_pred.values for y_pred in y_pred_list])

    # Check for NaN values in the input data
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        print(f"Warning: NaN values found in group {group_id}.")
        return  # Exit the function if NaNs are found

    mean_values = np.mean([y_true, y_pred], axis=0)
    diff_values = y_pred - y_true

    # Ensure diff_values is a NumPy array
    diff_values = np.array(diff_values)

    # Check for NaN values in differences
    if np.any(np.isnan(diff_values)):
        print(f"Warning: NaN values found in diff_values for group {group_id}.")
        return  # Exit the function if NaNs are found

    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    
    total_agreement_count = np.sum((diff_values >= lower_limit) & (diff_values <= upper_limit))
    total_agreement_percentage = (total_agreement_count / len(diff_values)) * 100


    # Create a colormap for the samples
    num_samples = len(y_true)
    # colors = cm.plasma(np.linspace(0, 1, num_samples))  # Use a more diverse colormap
    colors = cm.tab10(np.linspace(0, 1, num_samples))

    # Scatter plot with unique colors for each sample, using smaller dots
    ax.scatter(mean_values, diff_values, color=colors, s=20, alpha=0.7)  # s=20 for smaller dots

    ax.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
    ax.axhline(upper_limit, color='green', linestyle='--', label='Upper Limit of Agreement')
    ax.axhline(lower_limit, color='orange', linestyle='--', label='Lower Limit of Agreement')

    # Annotate the mean difference on the plot
    # ax.text(np.min(mean_values), mean_diff, f'{mean_diff:.2f}', color='black', fontsize=14, ha='left', va='center')
    
    ax.set_title(f'Bland-Altman Plot for Group {group_id} (Mean Diff: {mean_diff:.2f})')
    # equation = r'$(\hat{VO_2}+VO_2)/2$'
    equation = r'$(\widehat{VO}_2 + VO_2)/2$ '

    ax.set_xlabel(equation, fontsize=15)
    ax.set_ylabel(r'$\widehat{VO}_2 - VO_2$', fontsize=14)
    ax.tick_params(axis='both', labelsize=15)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    # ax.grid()
    
    #### 长宽比
    # ax.set_aspect(aspect=1.2)
    
def evaluate_model(model, loader, ori_df, ref_df, decoder_steps ,model_name, device):
    model.eval()
    for i, batch in enumerate(loader):
        # pdb.set_trace()
        with torch.no_grad():      
            outputs, attention_weights = model(batch)

        bs = batch["outputs"].shape[0]
        outputs = outputs.reshape(bs,decoder_steps,-1).cpu().detach().numpy()

        # pdb.set_trace()
        tmp = pd.DataFrame()

        tmp['VO2'] = outputs[:, :, 1].reshape(-1)


        # pdb.set_trace()
        tmp['ID'] = batch['identifier'].reshape(-1).tolist()
        rtmp = tmp[tmp['ID']!=0].reset_index(drop=True)
        
        ref_df['ID'] = ref_df['ID']-1
        out_df = ref_df.copy()
        out_df = out_df.drop(columns=['VO2'])
        out_df['VO2'] = rtmp['VO2']

        # pdb.set_trace()
        # Inverse trasformation
        real_columns = ['HR','HRR','VT','BF','VO2']
        categorical_columns = ['Status','Sex','Age','Height','Weight','WorkLoad']

        if model_name.endswith("base"):
            real_scalers, categorical_scalers = fit_preprocessing(ori_df, real_columns, categorical_columns=[])
            ref_df = inverse_transform_reals(ref_df, real_scalers, real_columns)
            true_df = ref_df
            out_df = inverse_transform_reals(out_df, real_scalers, real_columns)
        else:
            real_scalers, categorical_scalers = fit_preprocessing(ori_df, real_columns, categorical_columns)
            ref_df = inverse_transform_reals(ref_df, real_scalers, real_columns)
            ref_df = inverse_transform_categoricals(ref_df, categorical_scalers, categorical_columns)
            true_df = ref_df

            out_df = inverse_transform_reals(out_df, real_scalers, real_columns)
            out_df = inverse_transform_categoricals(out_df, categorical_scalers, categorical_columns)           


        return true_df, out_df


def compare_plot(data_dir,true_df, out_dfs, model_names, block):
    # plot_list=["TFCN","TFCN_base"]

    rec_ids = np.unique(true_df['ID'])

    test_df = pd.read_csv(os.path.join(data_dir,"./test.csv"),header=0)
    test_df = test_df.drop(['BMI'], axis=1)

    test_ids = test_df['ID'].unique()

    if block=='LSTM':
        train_save_path = "./TFN_results/LSTM/results_train"
        test_save_path = "./TFN_results/LSTM/results_test"
    elif block=='TCN':
        train_save_path = "./TFN_results/TCN/results_train"
        test_save_path = "./TFN_results/TCN/results_test"

    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    losses1 = []
    losses2 = []
    losses3 = []
    loss_test_1=[]
    loss_test_2=[]
    y_em_preds = []
    y_trues = []
    y_tests = []
    ### ADD for Bland plot
    bland_y_true=[]
    model_preds=[]
    base_model_preds=[]
    
    for e in rec_ids:
        # pdb.set_trace()
        plt.figure(figsize=(15,8))

        i = true_df[true_df["ID"]==e].index
        index = i.astype(int)
        index = [i-index[0] for i in index]
        index = [i for i in range(0,len(index)*2,2)]

        y_true = true_df[true_df["ID"]==e]['VO2']
        y_trues.append(y_true)
        y_preds = []
        # losses = []
        
        indexes = out_dfs[0][out_dfs[0]["ID"]==e].index
        ### y_preds[0]是model的预测， y_preds[1]是base_model的预测
        for out_df in out_dfs:
            # pdb.set_trace()
            y_pred = out_df.iloc[indexes]['VO2']
            y_preds.append(y_pred)
        
        

        sns.lineplot(x=index, y=y_true, color="black",label="ground truth", linewidth = 2.5 )

        colors = ["orange","purple"]
        # colors=["green","pink"]
        max_h = y_true.max()
        min_h = y_true.min()
        loss_id1 = []
        loss_id2 = []
        loss_id3 = []
        # criterion1 = nn.L1Loss()
        
        criterion2 = nn.MSELoss()
        criterion3 = nn.L1Loss()
 
        for i in range(len(y_preds)):
            # pdb.set_trace()
            # loss1 = criterion1(torch.tensor(y_preds[i].values),torch.tensor(y_true.values))
            loss1 = r2_score(torch.tensor(y_preds[i].values),torch.tensor(y_true.values))
            loss2 = criterion2(torch.tensor(y_preds[i].values),torch.tensor(y_true.values))    
            loss3 = criterion3(torch.tensor(y_preds[i].values),torch.tensor(y_true.values))
            loss_id1.append(loss1.item())
            loss_id2.append(loss2.item())
            loss_id3.append(loss3.item())
            ### 为了画图
            # model_names=["TFCN_base","TFCN"]
            model_names=["TFLSTM_base","TFLSTM"]
            ### ADD
            # bland_altman_plot(e,y_true, y_pred, model_names[i])
            
            if model_names[i]:
                sns.lineplot(x=index, y=y_preds[i], color=colors[i],label=model_names[i], linewidth = 3)
                max_h = max(y_preds[i].max(),max_h)
                min_h = min(y_preds[i].min(),min_h)
                plt.ylim(math.floor(min_h-1),math.ceil(max_h+0.1))
            

        
        plt.legend(fontsize=20)
        # pdb.set_trace()
        id_df = true_df[true_df['ID']==e]
        height = id_df['Height'].iloc[0]
        weight = id_df['Weight'].iloc[0]
        age = id_df['Age'].iloc[0]
        sex = id_df['Sex'].iloc[0]

        max_h = max(y_pred.max(),y_true.max())
        min_h = min(y_pred.min(),y_true.min())
        plt.ylim(0,math.ceil(max_h+0.1))
        if sex==1:
            sex_name="male"
        else:
            sex_name="female"
        plt.title(" Age: "+str(int(float(age)))+" Height: "+str(height)+" Weight: "+str(weight)+" Sex: "+sex_name, 
                  fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=24)


        # plt.fill_between(index, out_df[out_df["identifier"]==e].p10, out_df[out_df["identifier"]==e].p90, alpha=0.3)
        plt.ylabel("VO2 (L/min)",fontsize=26)
        plt.xlabel("Time (2 Seconds)",fontsize=26)
        
        # pdb.set_trace()
        
        if int(float(e)) in test_ids.astype(int):
            ### ADD
            bland_y_true.append(y_true)
            model_preds.append(y_preds[0])
            base_model_preds.append(y_preds[1])
            
            losses1.append(loss_id1)
            losses2.append(loss_id2)
            losses3.append(loss_id3)
            # plt.savefig(os.path.join(test_save_path,str(e[:-2])+"_line.png"))
            plt.savefig(os.path.join(test_save_path,str(e)+"_line.png"))
        else:
            # pdb.set_trace()
            # plt.savefig(os.path.join(train_save_path,str(e[:-2])+"_line.png"))
            plt.savefig(os.path.join(test_save_path,str(e)+"_line.png"))
        # plt.show()
        plt.close()
    
    
    
    
    ### ADD
    # print(test_ids)
    # pdb.set_trace()
    n_groups=4
    group_size=len(model_preds)//n_groups
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust grid size as needed
    axes = axes.flatten()  # Flatten to easily iterate over axes

    
    group_1 = [0, 3, 4, 6, 15, 16, 17, 18, 19, 26, 38] # avg:21
    group_2 = [1, 5, 7, 12, 23, 33, 40, 42, 37, 25, 28] # avg: 22
    group_3 = [10, 11, 9, 22, 35, 34, 13, 21, 27, 39, 14] # avg: 26.6
    group_4 = [36, 31, 41, 2, 29, 8, 32, 24, 8, 37, 20] # avg: 38.3

    groups = [group_1, group_2, group_3, group_4]

    # pdb.set_trace()
    ### ADD bland画图
    # for i, current_group in enumerate(groups):
    #     bland_altman_plot([bland_y_true[i] for i in current_group], [model_preds[i] for i in current_group], 
    #                       axes[i], group_id=i + 1)   

    #     plt.tight_layout()
    #     plt.savefig('bland_altman_faceted_plot.png')
        # plt.show()

    return losses1, losses2, losses3


def data_generator(X_train, y_train):
    while True:
        for i in range(len(X_train)):  # assuming X_train and y_train are lists of arrays
            yield np.expand_dims(X_train[i], axis=0), np.expand_dims(y_train[i], axis=0)


def main():
    with open('config.json','r') as f:
        params = json.load(f)
        
    block=params['block']
    data_dir="2s_data"
    # data_dir="3s_data"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE="cpu"
    
    
    
    test_file="./CPET_files.csv"
    trans_test_file="./Trans_CPET.csv"
    
    # test_file = "gaussian_low.csv"
    # trans_test_file="Trans_gaussian_low.csv"
    # test_file = "gaussian_mid.csv"
    # trans_test_file="Trans_gaussian_mid.csv"
    # test_file = "gaussian_high.csv"
    # trans_test_file="Trans_gaussian_high.csv"
    # test_file = "gaussian_mid_low.csv"
    # trans_test_file="Trans_gaussian_mid_low.csv"
    
    
    # test_file="./test_drift.csv"
    # trans_test_file="./Trans_test_drift.csv"
    # test_file="./test_jitter.csv"
    # trans_test_file="./Trans_test_jitter.csv"
    # test_file="./test_warp.csv"
    # trans_test_file="./Trans_test_warp.csv"
    # test_file="./test_gaussian.csv"
    # trans_test_file="./Trans_test_gaussian.csv"
    
    ori_df = pd.read_csv(os.path.join(data_dir,test_file),header=0) # the original file as the standard transformation
    ref_df = pd.read_csv(os.path.join(data_dir,trans_test_file),header=0) # the groundtruth file
    ori_df = ori_df.drop(['BMI','VE'], axis=1)
    ref_df = ref_df.drop(['BMI','VE'], axis=1)
    ### ADD
    ori_df['ID'] = pd.to_numeric(ori_df['ID'], errors='coerce') 
    ref_df['ID'] = pd.to_numeric(ref_df['ID'], errors='coerce') 


    input_columns = ["ID","Sex","Age","Status", "Height", "Weight","WorkLoad","HR","HRR","VT","BF"]
    base_input_columns = ["ID","HR","HRR","VT","BF"]
    target_column = ["VO2"]
    id_column = "ID"
    time_column = "Time"
    col_to_idx = {col: idx for idx, col in enumerate(input_columns)}
    base_col_to_idx = {col: idx for idx, col in enumerate(base_input_columns)}
    params['col_to_idx'] = col_to_idx
    base_params = params.copy()
    base_params['col_to_idx'] = base_col_to_idx
    ids_unique = ori_df['ID'].unique()
    weight_list = {}
    for id in ids_unique:
        # pdb.set_trace()
        indexes = ori_df[ori_df["ID"]==id].index
        weight_list[str(id)]=float(ori_df.iloc[indexes]['Weight'].iloc[0])
        
        
    if block=='TCN':
        # base:0 epoch: 50 swich: 25
        # base:1 epoch: 50 swich: 25
        model_names = ["TFCN","TFCN_base"]
        path_list = ["./models/tfcn_models","./models/tfcn_base_models"]
        checkpoints_list = [f"checkpoint_{44}.pth",f"checkpoint_{39}.pth"] 
    elif block=='LSTM':
        # base:0 epoch: 50  swich:25
        model_names = ["TFLSTM","TFLSTM_base"]
        path_list = ["./models/tflstm_models","./models/tflstm_base_models"]
        checkpoints_list = [f"checkpoint_{44}.pth",f"checkpoint_{29}.pth"] 
    
    # pdb.set_trace()
    
    model_list = []
    start_time = time.time()
    for model_name in model_names:
        if model_name == "TFCN" or model_name=="TFLSTM":
            model = TemporalFusionTransformer(params, DEVICE)
            model.to(DEVICE)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'{model_name}: Total number of parameters: {num_params}')
        elif model_name == "TFCN_base" or model_name=="TFLSTM_base":
            model = BaseTemporalFusionTransformer(base_params, DEVICE)
            model.to(DEVICE)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'{model_name}: Total number of parameters: {num_params}')

        model_list.append(model)
        



    decoder_steps=params['decoder_steps']

    #  Test on all the data including the training and testing
    testing_data = TFT_Dataset_test(ref_df, id_column, time_column, target_column, input_columns, decoder_steps)
    test_dataloader = DataLoader(testing_data, batch_size=500, num_workers=2, shuffle=False)
    
    # create the dataloader without static variables
    ref_df_base = ref_df.drop(columns=["WorkLoad","Sex","Age","Height","Weight","Status"])
    testing_data_base = TFT_Dataset_test(ref_df_base, id_column, time_column, target_column, base_input_columns, decoder_steps)
    test_dataloader_base = DataLoader(testing_data_base, batch_size=500, num_workers=2, shuffle=False)
    
    out_dfs = []
    ori_df_base = ori_df.drop(columns=["WorkLoad","Sex","Age","Height","Weight","Status"])
    for i,model in enumerate(model_list):
        # pdb.set_trace()
        model_path = path_list[i]
        checkpoint = torch.load(os.path.join(model_path,checkpoints_list[i]))
        model.load_state_dict(checkpoint['model_state_dict'])

        if model_names[i].endswith("base"):
            ref_df1 = ref_df_base.copy()
            true_base_df, out_df = evaluate_model(model, test_dataloader_base, ori_df_base, ref_df1, decoder_steps ,model_name=model_names[i], device=DEVICE)
        else:
            ref_df2 = ref_df.copy()
            true_df, out_df = evaluate_model(model, test_dataloader, ori_df, ref_df2, decoder_steps ,model_name=model_names[i], device=DEVICE)
        
        out_dfs.append(out_df)


    losses1, losses2, losses3= compare_plot(data_dir,true_df, out_dfs, model_names,block=params['block'])

    r2_loss = np.array(losses1).mean(axis=0)
    mse_loss = np.array(losses2).mean(axis=0)
    mae_loss = np.array(losses3).mean(axis=0)
    
    inference_time = time.time() - start_time
    print(f'Inference time: {inference_time:.2f} seconds')
    

    print("- "*10,'The block is ', params['block'],"- "*10)
    print(checkpoints_list)
    print("R error","--"*20)
    print(f"TFN: {r2_loss[0]:.3f}, TFN_base: {r2_loss[1]:.3f}")
    print("MSE error","--"*20)
    print(f"TFN: {mse_loss[0]:.3f}, TFN_base: {mse_loss[1]:.3f}")
    print("MAE error","--"*20)
    print(f"TFN: {mae_loss[0]:.3f}, TFN_base: {mae_loss[1]:.3f}")
    
    # Statistical tests
    # pdb.set_trace()
    # Convert the losses to a NumPy array for easier manipulation
    

    # for loss in [losses1,losses2,losses3]:
    #     losses_array = np.array(loss)

    #     # Extract the two sets of losses
    #     model_losses = losses_array[:, 0]
    #     base_model_losses = losses_array[:, 1]

    #     # Perform a paired t-test
    #     t_stat, p_value = stats.ttest_rel(model_losses, base_model_losses)

    #     # Print the results
    #     print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}") # 6.94647051901402e-08

    #     # Optional: Perform a Wilcoxon signed-rank test if the data is not normally distributed
    #     wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(model_losses, base_model_losses)
    #     print(f"Wilcoxon statistic: {wilcoxon_stat:.4f}, Wilcoxon P-value: {wilcoxon_p_value:.4f}") # 3.9473371228794425e-07

if __name__ == "__main__":
    main()


