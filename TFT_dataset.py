import numpy as np
from torch.utils.data import Dataset
#TODO
import pdb
import torch

class TFT_Dataset(Dataset):
    def __init__(self, data, id_column, time_column, target_column, 
                 input_columns, decoder_steps):
        """
          data (pd.DataFrame): dataframe containing raw data
          id_column (str): name of column containing entity data
          time_column (str): name of column containing date data
          target_column (str): name of column we need to predict
          input_columns (list): list of string names of columns used as input
          decoder_steps (int): number of input time steps used for each forecast date. Equivalent to the width N of the decoder
        """
        
        inputs = []
        outputs = []
        entity = []
        time = []
        
        for e in data[id_column].unique():
            entity_group = data[data[id_column] == e]
            data_time_steps = len(entity_group)
            
            if data_time_steps >= decoder_steps:
                x = entity_group[input_columns].values.astype(np.float32)
                y = entity_group[target_column].values.astype(np.float32)
                e = entity_group[[id_column]].values.astype(np.float32)
                t = entity_group[[time_column]].values.astype(np.int64)

                # Move the window by half the window length
                step_size = decoder_steps // 3
                
                
                # Create input windows
                for i in range(0, data_time_steps - decoder_steps + 1, step_size):
                    inputs.append(x[i:i + decoder_steps, :])
                    outputs.append(y[i:i + decoder_steps, np.newaxis])  # Add new axis for outputs
                    entity.append(e[i:i + decoder_steps, :])
                    time.append(t[i:i + decoder_steps, :])

        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.entity = np.array(entity)
        self.time = np.array(time)

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'time': self.time,
            'identifier': self.entity
        }
        
    def __getitem__(self, index):
        return {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
            'time': self.time[index],
            'identifier': self.entity[index]
        }

    def __len__(self):
        return self.inputs.shape[0]