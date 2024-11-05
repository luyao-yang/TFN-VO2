import numpy as np
from torch.utils.data import Dataset
#TODO
import pdb
import torch

class TFT_Dataset_test(Dataset):
    def __init__(self, data, id_column, time_column, target_column, 
                 input_columns, decoder_steps):
        """
          data (pd.DataFrame): dataframe containing raw data
          entity_column (str): name of column containing entity data
          time_column (str): name of column containing date data
          target_column (str): name of column we need to predict
          input_columns (list): list of string names of columns used as input
          encoder_steps (int): number of known past time steps used for forecast. Equivalent to size of LSTM encoder
          decoder_steps (int): number of input time steps used for each forecast date. Equivalent to the width N of the decoder
        """
        
        # self.encoder_steps = encoder_steps
             
        inputs = []
        outputs = []
        entity = []
        time = []
        
        
        # 对于16个entity，不停的遍历每个entity，每个entity都是一个user id
        # encoder_steps: 175  decoder_step: 180
        # TODO 整数处理机制是什么样的
        for e in data[id_column].unique():
          
            entity_group = data[data[id_column]==e]

            data_time_steps = len(entity_group) # 对于每一个id，有多少个时间点，时间细粒度的数量 

            # pdb.set_trace()
            x = entity_group[input_columns].values.astype(np.float32)
            y = entity_group[target_column].values.astype(np.float32)
            e = entity_group[id_column].values.astype(np.float32)
            t = entity_group[time_column].values.astype(np.int64)
            
            if data_time_steps < decoder_steps:
                tmp = np.zeros((decoder_steps,len(input_columns)))
                tmp[:len(x),:]=x
                inputs.append(tmp)

                otmp = np.zeros((decoder_steps,len(target_column)))
                otmp[:len(y),:]=y
                outputs.append(otmp)

                ttmp = np.zeros((decoder_steps,))
                ttmp[:len(t),]=t
                time.append(ttmp)

                etmp = np.zeros((decoder_steps,))
                etmp[:len(e),]=e
                entity.append(etmp)


            if data_time_steps >= decoder_steps:
                div = data_time_steps // decoder_steps
                mode = data_time_steps % decoder_steps
                for i in range(div):
                    start_i = i*decoder_steps
                    inputs.append(x[start_i:start_i+decoder_steps,:])
                    outputs.append(y[start_i:start_i+decoder_steps,:])
                    entity.append(e[start_i:start_i+decoder_steps,])
                    time.append(t[start_i:start_i+decoder_steps,])
                if mode:
                    tmp=np.zeros((decoder_steps, len(input_columns)))
                    tmp[:mode,:]=x[-mode:,:]
                    inputs.append(tmp)

                    otmp=np.zeros((decoder_steps,len(target_column)))
                    otmp[:mode,:]=y[-mode:,:]
                    outputs.append(otmp)

                    etmp=np.zeros((decoder_steps,))
                    etmp[:mode,]=e[-mode:,]
                    entity.append(etmp)

                    ttmp=np.zeros((decoder_steps,))
                    ttmp[:mode,]=t[-mode:,]
                    time.append(ttmp)

        # pdb.set_trace()
        # self.inputs = np.stack(inputs, axis=0)
        self.inputs = torch.from_numpy(np.stack(inputs, axis=0))
        # self.outputs = np.concatenate(outputs, axis=0)[:,encoder_steps:,:]
        self.outputs = torch.from_numpy(np.stack(outputs, axis=0))
        self.entity = torch.from_numpy(np.stack(entity, axis=0)).unsqueeze(-1)
        self.time = torch.from_numpy(np.stack(time, axis=0)).unsqueeze(-1)
        # self.entity = e
        # self.time = t
        # self.active_inputs = np.ones_like(outputs)
        # print(self.inputs.shape) # (14672, 180, 10)
     

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs,
            # 'active_entries': np.ones_like(self.outputs),
            'time': self.time,
            'identifier': self.entity
        }
        
    def __getitem__(self, index):
        s = {
        'inputs': self.inputs[index],
        'outputs': self.outputs[index], 
        # 'active_entries': np.ones_like(self.outputs[index]), 
        'time': self.time[index],
        'identifier': self.entity[index]
        }

        return s


    def __len__(self):
        return self.inputs.shape[0]