import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pdb


# StandardScaler is used to normalize the continuous variables (removing the mean and sciling to unit variance)
#  LabelEncoder is used to transform categhorical variables into integers
def fit_preprocessing(train, real_columns, categorical_columns):
    real_scalers = StandardScaler().fit(train[real_columns].values)

    categorical_scalers = {}
    num_classes = []
    for col in categorical_columns:
        srs = train[col].apply(str) 
        categorical_scalers[col] = LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique())

    return real_scalers, categorical_scalers


def transform_inputs(df, real_scalers, categorical_scalers, real_columns, categorical_columns):
    out = df.copy()
    out[real_columns] = real_scalers.transform(df[real_columns].values)

    # pdb.set_trace()
    for col in categorical_columns:
        # string_df = df[col].apply(str)
        out[col] = categorical_scalers[col].transform(df[col].astype(str))

    return out

def inverse_transform_reals(df, real_scalers, real_columns):
    out = df.copy()
    out[real_columns] = real_scalers.inverse_transform(df[real_columns].values)
    return out

def inverse_transform_categoricals(df, categorical_scalers, categorical_columns):
    # pdb.set_trace()
    out = df.copy()
    for col in categorical_columns:
        out[col] = categorical_scalers[col].inverse_transform(df[col].values)
    return out


def data_preprocess(folder_path, data_save_path):
    # Set the total dataframe
    CPET_files = [] 

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Read the time series from an Excel file
        # df = pd.read_excel(file_path, parse_dates=['Time'])
        # df = pd.read_excel(file_path,header=0, engine="openpyxl")
        # print(file_path)
        df = pd.read_csv(file_path, index_col=False)
        # df=df.drop(['RER','VCO2'],axis=1)

        # print(file_path)
        # print(np.unique(df['Age']))
        expected_columns=15
        assert len(df.columns) == expected_columns, f"{file} number of columns is not {expected_columns}"
        # print(file,": ",df["VO2"].iloc[0])
        CPET_files.append(df)
        

    # pdb.set_trace()
    final_CPET_files = pd.concat(CPET_files)
    final_CPET_files.reset_index(drop=True, inplace=True)
    save_file_path = os.path.join(data_save_path, "CPET_files.csv")
    final_CPET_files.to_csv(save_file_path,index=False)


def plot_CPET(CPET_files):

    rec_ids = CPET_files['ID'].unique()
    # pdb.set_trace()
    
    for e in rec_ids:
        file_id = CPET_files[CPET_files["ID"]==e]
        plt.figure(figsize=(20,10))
        i = file_id['Time']
        index= [0]
        # pdb.set_trace()
        for x in i[1:]:
            t = int(x)
            index.append(t)
        index = [i-index[0] for i in index]
        
        sns.lineplot(x=index, y=file_id['VO2'], color="black",label="ground truth")
        plt.tick_params(axis='both', which='major', labelsize=14)

        if not os.path.exists("./truth_plots"):
            os.makedirs("./truth_plots")
        plt.savefig(os.path.join("./truth_plots",str(e)+".png"))
        plt.close()


def plot_box_group(CPET_files,var):
    # pdb.set_trace()
    group_ids = CPET_files[var].unique()
    plt.figure(figsize=(8,6))
    if var=="WorkLoad":
        labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        bins = [-1, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330,350,370,390]
        ages = group_ids
        groups = pd.cut(group_ids, bins=bins, labels=labels)
        groups_dict = {label: [] for label in labels}
        
        # pdb.set_trace()
        for age, group in zip(ages, groups):
            groups_dict[group].append(age)
        groups_dict = {k: v for k, v in groups_dict.items() if v}

        data = []
        for group, group_ids in groups_dict.items():
            mean_group =[np.mean(CPET_files[CPET_files[var]==id]['VO2']) for id in group_ids]
            for mean in mean_group:
                data.append({'Group': group, 'VO2': mean})

        # Create a DataFrame
        # pdb.set_trace()
        df = pd.DataFrame(data)

        # Create a boxplot using seaborn
        sns.boxplot(x='Group', y='VO2', data=df)
        plt.xlabel('WorkLoad Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)

        plt.tick_params(axis='both', which='major', labelsize=14)


        plt.title('Boxplot for WorkLoad groups', fontsize=18)

    if var=="BMI":
        labels = [1,2,3,4]
        bins = [15, 20, 25, 30, 40]
        ages = group_ids
        groups = pd.cut(group_ids, bins=bins, labels=labels)
        groups_dict = {label: [] for label in labels}
        
        for age, group in zip(ages, groups):
            groups_dict[group].append(age)
        groups_dict = {k: v for k, v in groups_dict.items() if v}

        data = []
        for group, group_ids in groups_dict.items():
            mean_group =[np.mean(CPET_files[CPET_files[var]==id]['VO2']) for id in group_ids]
            for mean in mean_group:
                data.append({'Group': group, 'VO2': mean})

        # Create a DataFrame
        # pdb.set_trace()
        df = pd.DataFrame(data)


        # Create a boxplot using seaborn
        colors = ['steelblue', 'orange', 'forestgreen', 'firebrick']
        sns.boxplot(x='Group', y='VO2', data=df, palette=colors)

        plt.xlabel('BMI Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)

        # Set x-axis labels
        plt.xticks(ticks = [0,1,2,3],labels=['15-20', '20-25', '25-30',"30-36"]) # replace with your labels
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title('Boxplot for BMI groups',fontsize=18)

    if var=="Weight":
        labels = [1,2,3,4,5,6]
        bins = [50, 60, 70, 80, 90, 100,130]
        ages = group_ids
        groups = pd.cut(group_ids, bins=bins, labels=labels)
        groups_dict = {label: [] for label in labels}
        
        for age, group in zip(ages, groups):
            groups_dict[group].append(age)
        groups_dict = {k: v for k, v in groups_dict.items() if v}

        data = []
        for group, group_ids in groups_dict.items():
            mean_group =[np.mean(CPET_files[CPET_files[var]==id]['VO2']) for id in group_ids]
            for mean in mean_group:
                data.append({'Group': group, 'VO2': mean})

        # Create a DataFrame
        # pdb.set_trace()
        df = pd.DataFrame(data)

        # Create a boxplot using seaborn
        colors = ['steelblue', 'orange', 'forestgreen', 'firebrick', 'purple', 'sienna', 'hotpink']
        sns.boxplot(x='Group', y='VO2', data=df, palette=colors)

        plt.xlabel('Weight Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)

        # Set x-axis labels
        plt.xticks(ticks = [0,1,2,3,4,5],labels=['50-60', '60-70', '70-80',"80-90","90-100","100-120"]) # replace with your labels
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title('Boxplot for weight groups',fontsize=18)

    if var=="Height":
        labels = [1,2,3,4]
        bins = [157, 170, 180, 190, 200]
        ages = group_ids
        groups = pd.cut(group_ids, bins=bins, labels=labels)
        groups_dict = {label: [] for label in labels}
        
        for age, group in zip(ages, groups):
            groups_dict[group].append(age)
        groups_dict = {k: v for k, v in groups_dict.items() if v}

        data = []
        for group, group_ids in groups_dict.items():
            mean_group =[np.mean(CPET_files[CPET_files[var]==id]['VO2']) for id in group_ids]
            for mean in mean_group:
                data.append({'Group': group, 'VO2': mean})

        # Create a DataFrame
        # pdb.set_trace()
        df = pd.DataFrame(data)

        # Create a boxplot using seaborn
        
        colors=['firebrick', 'purple', 'sienna', 'hotpink']
        sns.boxplot(x='Group', y='VO2', data=df,palette=colors)
        plt.xlabel('Height Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)

        # Set x-axis labels
        plt.xticks(ticks = [0,1,2,3],labels=['1.58-1.7', '1.7-1.8', '1.8-1.9',"1.9-2.0"]) # replace with your labels
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title('Boxplot for height groups',fontsize=18)

    if var=="Age":
        # pdb.set_trace()
        # if file=="97.xlsx.csv":
        #     pdb.set_trace()
        labels = [1,2,3,4,5,6,7,8,9,10]
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ages = group_ids
        groups = pd.cut(group_ids, bins=bins, labels=labels)
        groups_dict = {label: [] for label in labels}
        
        # pdb.set_trace()
        for age, group in zip(ages, groups):
            # print(ages,group)
            groups_dict[group].append(age)
        groups_dict = {k: v for k, v in groups_dict.items() if v}
        
        # file_path="group_dict.txt"
        # with open(file_path, "w") as file:
        #     file.write(str(groups_dict))
        # groups_dict

        data = []
        
        for group, group_ids in groups_dict.items():
            mean_group =[np.mean(CPET_files[CPET_files[var]==id]['VO2']) for id in group_ids]
            for mean in mean_group:
                data.append({'Group': group, 'VO2': mean})

        # pdb.set_trace()
        # Create a DataFrame
        df = pd.DataFrame(data)

        # Create a boxplot using seaborn
        colors = ['steelblue', 'orange', 'forestgreen', 'firebrick']
        # colors = ['firebrick', 'purple', 'sienna', 'hotpink']
        sns.boxplot(x='Group', y='VO2', data=df, palette=colors)

        plt.xlabel('Age Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)

        # Set x-axis labels
        plt.xticks(ticks = [0,1,2,3],labels=['20-30', '30-40', '40-50','50-60']) # replace with your labels
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title('Boxplot for age groups',fontsize=18)
  
    if var=="Sex":
        # pdb.set_trace()
        # Filter for only Summer and Winter
        df = CPET_files[CPET_files['Sex'].isin([1, 0])]
        # Now use seaborn to create a boxplot
        colors = ['steelblue', 'orange']
        sns.boxplot(x=df['Sex'], y=df['VO2'],palette=colors)

        plt.xlabel('Sex Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)
        # plt.boxplot(x=df['Sex'], y=df['VO2'])
        plt.title('Boxplot for sex groups',fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        
    
    if var=="Status":
            # pdb.set_trace()
        # Filter for only Summer and Winter
        df = CPET_files[CPET_files['Status'].isin([1, 0])]
        # Now use seaborn to create a boxplot
        colors=['purple', 'sienna']
        sns.boxplot(x=df['Status'], y=df['VO2'],palette=colors)

        plt.xlabel('Status Group', fontsize=18)  # Change the fontsize to your desired size
        plt.ylabel('VO2 (L/min)', fontsize=18)
        # plt.boxplot(x=df['Sex'], y=df['VO2'])
        plt.title('Boxplot for status groups',fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        


    if not os.path.exists("./figs"):
        os.makedirs("./figs")
    plt.savefig(os.path.join("./figs",var+"_bp.png"))
    plt.close()
    # pdb.set_trace()


def seperate_train_test(CPET_files, CPET_files_ori,data_save_path):
    # pdb.set_trace()
    
    labels = [1,2,3,4,5,6,7,8,9,10]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    group_ids = CPET_files_ori.groupby(["ID","Age"])["ID","Age"].apply(lambda x: x.drop_duplicates(["ID", "Age"]))
    # group_ids = group_ids["Age"]
    groups = pd.cut(group_ids["Age"], bins=bins, labels=labels)
    groups_dict = {label: [] for label in labels}
    
    # pdb.set_trace()
    for row, group in zip(group_ids.iterrows(), groups):
        # print(ages,group)
        age=row[1]["Age"]
        id=row[1]["ID"]
        groups_dict[group].append(id)
    age_groups = {k: v for k, v in groups_dict.items() if v}
    
    # pdb.set_trace()
    # Separate IDs and age groups
    ids = []
    age_labels = []
    for age_group, id_list in age_groups.items():
        ids.extend(id_list)
        age_labels.extend([age_group] * len(id_list))

    # pdb.set_trace()
    # Stratified train-test split within each age group
    train_ids, test_ids, train_age_labels, test_age_labels = train_test_split(
        ids, age_labels, test_size=0.3, stratify=age_labels, random_state=32
    )


    # pdb.set_trace()
    # train_ids, test_ids = train_test_split(CPET_files['ID'], test_size=0.3, stratify=groups,random_state=32)
    train_df = CPET_files[CPET_files['ID'].isin(train_ids)]
    test_df = CPET_files[CPET_files['ID'].isin(test_ids)]
    
    # print(len(train_ids))
    # print("Training IDs:", train_ids)
    # print(len(test_ids))
    # print("Testing IDs:", test_ids)
    
    # print()
    # train_df, test_df = train_test_split(CPET_files, test_size=0.3, stratify=CPET_files['Age'])


    # train_df.to_csv(os.path.join(data_save_path,"./train.csv"),index=None)
    # test_df.to_csv(os.path.join(data_save_path,"./test.csv"),index=None)

def main():

    # read_file = "CPET_files.csv"
    # save_trans_file="Trans_CPET.csv"
    
    # read_file = "test_drift.csv"
    # save_trans_file="Trans_test_drift.csv"
    # read_file = "test_jitter.csv"
    # save_trans_file="Trans_test_jitter.csv"
    # read_file = "test_gaussian.csv"
    # save_trans_file="Trans_test_gaussian.csv"
    # read_file = "test_warp.csv"
    # save_trans_file="Trans_test_warp.csv"
    
    # read_file = "gaussian_low.csv"
    # save_trans_file="Trans_gaussian_low.csv"
    # read_file = "gaussian_mid.csv"
    # save_trans_file="Trans_gaussian_mid.csv"
    # read_file = "gaussian_high.csv"
    # save_trans_file="Trans_gaussian_high.csv"
    read_file = "gaussian_mid_low.csv"
    save_trans_file="Trans_gaussian_mid_low.csv"
    
    
    folder_path = "./2s"
    data_save_path = "./2s_data"
    # data_save_path = "./3s_data"
    ## Dara preprocess
    data_preprocess(folder_path, data_save_path)

    real_columns = ['HR','HRR','VE','VT','BF','VO2']
    categorical_columns = ['ID','Sex','Age','Height','Weight','BMI','WorkLoad','Status']

    # pdb.set_trace()
    CPET_files = pd.read_csv(os.path.join(data_save_path,read_file))
    # CPET_files = pd.read_csv(os.path.join(data_save_path,"CPET_files.csv"))

    # plot_box_group(CPET_files,"Age")
    # plot_box_group(CPET_files,"BMI")
    # plot_box_group(CPET_files,"Sex")
    # plot_box_group(CPET_files,"Height")
    # plot_box_group(CPET_files,"Weight")
    # plot_box_group(CPET_files,"Status")
    # plot_box_group(CPET_files,"WorkLoad")


    # Transformation
    real_scalers, categorical_scalers = fit_preprocessing(CPET_files, real_columns, categorical_columns)
    CPET_trans_files = transform_inputs(CPET_files, real_scalers, categorical_scalers, real_columns, categorical_columns)
    # pdb.set_trace()
    CPET_trans_files['ID'] = CPET_trans_files['ID']+1
    CPET_trans_files.to_csv(os.path.join(data_save_path,save_trans_file), index=None)
    
    # plot_CPET(CPET_files)

    print("******* Statistics for CPETs ************")
    print("Height mean: ", CPET_files["Height"].mean(), "   Var: ",CPET_files["Height"].std(), "  Num", len(CPET_files["Height"].unique()))
    print("Height mean: ", CPET_files["Weight"].mean(), "   Var: ",CPET_files["Weight"].std(), "  Num", len(CPET_files["Weight"].unique()))
    print("Age mean: ", CPET_files["Age"].mean(), "   Var: ",CPET_files["Age"].std(), "  Num", len(CPET_files["Age"].unique()))
    print("BMI mean: ", CPET_files["BMI"].mean(), "   Var: ",CPET_files["BMI"].std(), "  Num", len(CPET_files["BMI"].unique()))
    print("VO2 mean: ", CPET_files["VO2"].mean(), "   Var: ",CPET_files["VO2"].std())
    vo2_max=CPET_files["VO2"].max
    # counts = ['Sex'].value_counts()

    female_count = CPET_files[CPET_files['Sex'] == 0]['ID'].nunique()
    male_count = CPET_files[CPET_files['Sex'] == 1]['ID'].nunique()
    healthy_count = CPET_files[CPET_files['Status'] == 0]['ID'].nunique()
    smoker_count = CPET_files[CPET_files['Status'] == 1]['ID'].nunique()
    


    print("Female count:", female_count)
    print("Male count:", male_count)
    print("Health count:", healthy_count)
    print("Smoker count:", smoker_count)
    # pdb.set_trace()
    # Use the files after transformation and further denosing step
    # seperate_train_test(CPET_trans_files, CPET_files,data_save_path)
    
    ids=CPET_files['ID'].unique()
    max_values=[]
    for id in ids:
        max_values.append(CPET_files[CPET_files['ID']==id]['VO2'].max())
    mean = sum(max_values) / len(max_values)

    # Calculate the standard deviation
    std = (sum((x - mean) ** 2 for x in max_values) / len(max_values)) ** 0.5
    print("mean VO2 max: ", mean, std)


# ******* Statistics for CPETs ************
# Height mean:  175.94043696926113    Var:  8.524725730615668   Num 66
# Weight mean:  75.72925033711105    Var:  13.019646500118377   Num 111
# Age mean:  26.576867779435833    Var:  7.213130615324958   Num 51
# BMI mean:  24.364076310899993    Var:  3.0755054461063165   Num 138
# VO2 mean:  1.6868396746728735    Var:  0.8628437984873164
# Female count: 52
# Male count: 92
# Health count: 79
# Smoker count: 65

if __name__=="__main__":
    main()




