import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pdb
# data_save_path="./pack0"
CPET_files = [] 

# if not os.path.exists(data_save_path):
#     os.makedirs(data_save_path)

# folder_path="./original_packs/pack5"
# folder_path="./original_packs/pack4"
# folder_path="./original_packs/pack3"
# folder_path="./original_packs/pack2"
folder_path="./original_packs/pack1"
# folder_path="./original_packs/pack0"

interval='2S'
# interval='3S'


def convert_to_seconds(time_string):
    # pdb.set_trace()
    # if file=="105.xlsx":
        # pdb.set_trace()  
        # print(time_string)
    if folder_path=="./original_packs/pack5" or folder_path=="./original_packs/pack4":
        hour, minutes, seconds = time_string.split(":")
    else:
        minutes, seconds = time_string.split(":")
    minutes = int(minutes)
    seconds = int(float(seconds))
    total_seconds = minutes * 60 + seconds
    return total_seconds

# Apply the conversion function to the 'Time' column
# df['Time'] = df['Time'].apply(convert_to_seconds)

for file in os.listdir(folder_path):
    print(file)
    file_path = os.path.join(folder_path, file)

    # Read the time series from an Excel file
    # df = pd.read_excel(file_path, parse_dates=['Time'])
    # pdb.set_trace()
    df = pd.read_excel(file_path,header=0, engine="openpyxl")

    df['ID']=df['ID'][0]
    df['Height']=df['Height'][0]
    df['Weight']=df['Weight'][0]
    df['Age']=df['Age'][0]
    df['Sex']=df['Sex'][0]
    df['Status']=df['Status'][0]
    df['BMI']=df['Weight'][0]/(df['Height'][0]*df['Height'][0])*10000
    df['BF'] = df['BF'].astype(float)
    df['VCO2'] = df['VCO2'].astype(float)
    if folder_path=="./original_packs/pack1" or folder_path=="./original_packs/pack2" or folder_path=="./original_packs/pack3":
        df['VCO2'] = df['VCO2']/1000
        df['VO2'] = df['VO2']/1000
        
    
    if folder_path!="./original_packs/pack0":
        df['Time'] = df['Time'].astype(str)
        

    
    #### AS for pack5
    if folder_path=="./original_packs/pack5":
        df = df.drop(['BSA', 'Humidity','Temperature','VO2/kg','RER','VCO2'], axis=1)
    else:
        df = df.drop(['RER','VCO2'], axis=1)
    if file=="25.xlsx":
        df = df.drop(['  min   '], axis=1)
        
    expected_columns=15
    assert len(df.columns) == expected_columns, f"{file} number of columns is not {expected_columns}"
        
    # df['Time'] = df['Time'].astype(int)
        
    rest_avg = np.nanmean(df[df['WorkLoad'] == 0]['HR'])
    max_value = np.max(df['HR'])
    df['HRR'] = (df['HR']-rest_avg)/(max_value-rest_avg)
    # Pack2
    if file != "83.xlsx" and folder_path!="./original_packs/pack0":
        df['Time'] = df['Time'].apply(convert_to_seconds)
    else:
        df['Time'] = df['Time'].astype(float)

        
        
 
    df['Time'] = df['Time'] - df['Time'].min() 
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    # pdb.set_trace()
    df.set_index('Time', inplace=True)
    
    # if file=="35.xlsx":
    #     pdb.set_trace()
    
    # pdb.set_trace()
    sex = df["Sex"].iloc[0]
    df = df.resample(interval).mean().interpolate(method="linear")
    df.index = (df.index - pd.Timestamp('1970-01-01 00:00:00')).total_seconds()
    df = df.iloc[8:].dropna()
    
    
    if sex=="male":
        df["Sex"] = 1
    elif sex=="female":
        df["Sex"] = 0
    
    
    
    workloads = df['WorkLoad'].unique()
    

        
    for workload in workloads:
        for col in ['VO2', 'HR', 'HRR', 'VE', 'VT', 'BF']:
            # if file=="41.xlsx" and col=="VO2":
            #     pdb.set_trace()
            # Calculate mean and standard deviation
            tmp = df[df['WorkLoad']==workload][col]
            mean = tmp.mean(skipna=True)
            std = tmp.std(skipna=True)

            # pdb.set_trace()
            
            # Create boolean conditions for values to be removed
            condition = (tmp > mean + 2 * std) | (tmp< mean - 2 * std)

            # Remove values that satisfy the condition
            df.loc[df['WorkLoad'] == workload, col] = tmp.loc[~condition]
            

            
            if folder_path=="./original_packs/pack1" or folder_path=="./original_packs/pack2" or folder_path=="./original_packs/pack3":
                tmp = df[df['WorkLoad']==workload][col]
                mean = tmp.mean(skipna=True)
                std = tmp.std(skipna=True)

                # pdb.set_trace()
                
                # Create boolean conditions for values to be removed
                condition = (tmp > mean + 2 * std) | (tmp< mean - 2 * std)

                # Remove values that satisfy the condition
                df.loc[df['WorkLoad'] == workload, col] = tmp.loc[~condition]
                
                            ### The second timen
            
                tmp = df[df['WorkLoad']==workload][col]
                mean = tmp.mean(skipna=True)
                std = tmp.std(skipna=True)

                # pdb.set_trace()
                
                # Create boolean conditions for values to be removed
                condition = (tmp > mean + 2 * std) | (tmp< mean - 2 * std)

                # Remove values that satisfy the condition
                df.loc[df['WorkLoad'] == workload, col] = tmp.loc[~condition]
            
            if file in ["97.xlsx","82.xlsx","79.xlsx","76.xlsx","75.xlsx","71.xlsx","66.xlsx","67.xlsx","62.xlsx","106.xlsx","106.xlsx","99.xlsx","101.xlsx","105.xlsx","108.xlsx",
                        "109.xlsx","6.xlsx","8.xlsx","12.xlsx","21.xlsx","37.xlsx","43.xlsx","45.xlsx","110.xlsx","111.xlsx","112.xlsx","119.xlsx","122.xlsx","136.xlsx"]:
                tmp = df[df['WorkLoad']==workload][col]
                mean = tmp.mean(skipna=True)
                std = tmp.std(skipna=True)

                # pdb.set_trace()
                
                # Create boolean conditions for values to be removed
                condition = (tmp > mean + 2 * std) | (tmp< mean - 2 * std)

                # Remove values that satisfy the condition
                df.loc[df['WorkLoad'] == workload, col] = tmp.loc[~condition]
            
            if file in ["6.xlsx","8.xlsx","12.xlsx","97.xlsx","82.xlsx","79.xlsx","76.xlsx","75.xlsx","71.xlsx","67.xlsx","62.xlsx","21.xlsx","109.xlsx","111.xlsx","119.xlsx",
                        "37.xlsx","43.xlsx","112.xlsx","122.xlsx"]:
                tmp = df[df['WorkLoad']==workload][col]
                mean = tmp.mean(skipna=True)
                std = tmp.std(skipna=True)

                # pdb.set_trace()
                
                # Create boolean conditions for values to be removed
                condition = (tmp > mean + 2 * std) | (tmp< mean - 2 * std)

                # Remove values that satisfy the condition
                df.loc[df['WorkLoad'] == workload, col] = tmp.loc[~condition]
                
            
    # pdb.set_trace()    
    # if file=="35.xlsx":
    #     pdb.set_trace()
    
    df = df.iloc[4:].dropna()
    df['ID']=df['ID'].iloc[0]
    df['Height']=df['Height'].iloc[0]
    df['Weight']=df['Weight'].iloc[0]
    df['Age']=df['Age'].iloc[0]
    df['Sex']=df['Sex'].iloc[0]
    df['Status']=df['Status'].iloc[0]
    # pdb.set_trace()
    df['BMI']=df['BMI'].iloc[0]
    # df['Time'] = df['Time'] - df['Time'].min() 

    df=df.interpolate()
    df['Time']=df.index
    df['Time'] = df['Time'] - df['Time'].min() 
    
    expected_columns=15
    assert len(df.columns) == expected_columns, f"{file} number of columns is not {expected_columns}" 
    
    bins = [-1, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330,350,370,390]
    # pdb.set_trace()
    df['WorkLoad'] = pd.cut(df['WorkLoad'], bins=bins, labels=np.arange(0,400,20))
    
    df.reset_index(drop=True, inplace=True)
    # dir="pack5_new"
    if interval=='2S':
        dir="2s/"
    else:
        dir="3s/"
        
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_file_path = os.path.join(dir,file+".csv")
    df.to_csv(save_file_path,index=False)

    

