import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn import model_selection


class IrregularTimeSeries(object):
    def __init__(self, root, dataset_name, n_samples=None, device=torch.device("cpu")):
        self.root = root
        self.dataset_name = dataset_name
        self.device = device
        
        self.data_file = f"{dataset_name}.csv"
        self.data_path = os.path.join(root, self.data_file)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = self._process_data()
        
        if n_samples is not None:
            print(f'Total records: {len(self.data)}')
            self.data = self.data[:n_samples]
            print(f'Limited records: {len(self.data)}')
    
    def _process_data(self):
        print(f'Processing {self.data_file}...')
        
        df = pd.read_csv(self.data_path)
        
        date_col = df.columns[0]
        feature_cols = df.columns[1:]
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        feature_values = df[feature_cols].values.astype(np.float32)
        feature_tensor = torch.tensor(feature_values).to(self.device)
        
        mask_tensor = (feature_tensor != 0.0).float()
        
        window_size = 200
        stride = 50
        
        records = []
        total_length = len(df)
        
        for i in range(0, total_length - window_size + 1, stride):
            start_idx = i
            end_idx = min(i + window_size, total_length)
            
            time_steps = torch.arange(end_idx - start_idx, dtype=torch.float32).to(self.device)
            
            window_features = feature_tensor[start_idx:end_idx]
            window_mask = mask_tensor[start_idx:end_idx]
            
            record_id = f"{self.dataset_name}_sample_{i//stride}"
            
            records.append((record_id, time_steps, window_features, window_mask))
        
        print(f'Created {len(records)} time series samples')
        return records
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dataset name: {}\n'.format(self.dataset_name)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def irregular_timeseries_time_chunk(data, args, device):
    chunk_data = []
    history = args.history
    pred_window = args.pred_window
    
    for b, (record_id, tt, vals, mask) in enumerate(data):
        t_max = int(tt.max())
        
        for st in range(0, t_max - history, pred_window):
            et = st + history + pred_window
            if et >= t_max:
                idx = torch.where((tt >= st) & (tt <= et))[0]
            else:
                idx = torch.where((tt >= st) & (tt < et))[0]
            
            new_id = f"{record_id}_{st//pred_window}"
            chunk_data.append((new_id, tt[idx] - st, vals[idx], mask[idx]))
    
    return chunk_data


def irregular_timeseries_get_seq_length(args, records):
    max_input_len = 0
    max_pred_len = 0
    lens = []
    
    for b, (record_id, tt, vals, mask) in enumerate(records):
        n_observed_tp = torch.lt(tt, args.history).sum()
        max_input_len = max(max_input_len, n_observed_tp)
        max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
        lens.append(n_observed_tp)
    
    lens = torch.stack(lens, dim=0)
    median_len = lens.median()
    
    return max_input_len, max_pred_len, median_len


def irregular_timeseries_no_padding_variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train", 
                                                            data_min=None, data_max=None, time_max=None):
    processed_batch = []
    
    for b, (record_id, tt, vals, mask) in enumerate(batch):
        n_observed_tp = torch.lt(tt, args.history).sum()
        
        observed_tp = tt[:n_observed_tp]
        observed_data = vals[:n_observed_tp]
        observed_mask = mask[:n_observed_tp]
        
        predicted_tp = tt[n_observed_tp:]
        predicted_data = vals[n_observed_tp:]
        predicted_mask = mask[n_observed_tp:]
        
        observed_data = normalize_masked_data(observed_data.unsqueeze(0), observed_mask.unsqueeze(0), 
                                             att_min=data_min, att_max=data_max).squeeze(0)
        predicted_data = normalize_masked_data(predicted_data.unsqueeze(0), predicted_mask.unsqueeze(0), 
                                              att_min=data_min, att_max=data_max).squeeze(0)
        
        observed_tp = normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
        predicted_tp = normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)
        
        data_dict = {
            "observed_data": observed_data,
            "observed_tp": observed_tp,
            "observed_mask": observed_mask,
            "data_to_predict": predicted_data,
            "tp_to_predict": predicted_tp,
            "mask_predicted_data": predicted_mask,
            "record_id": record_id
        }
        
        processed_batch.append(data_dict)
    
    return processed_batch


def normalize_masked_data(data, mask, att_min=None, att_max=None):
    if att_min is None:
        att_min = torch.min(data)
    if att_max is None:
        att_max = torch.max(data)
    
    range_val = att_max - att_min
    if range_val > 0:
        data = (data - att_min) / range_val
    return data


def normalize_masked_tp(data, att_min=None, att_max=None):
    if att_min is None:
        att_min = torch.min(data)
    if att_max is None:
        att_max = torch.max(data)
    
    range_val = att_max - att_min
    if range_val > 0:
        data = (data - att_min) / range_val
    return data
