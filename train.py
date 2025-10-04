import os
import sys
import time
import datetime
import argparse
import numpy as np
import random
from random import SystemRandom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model.SEDformer import SEDformer
from utils.irregular_timeseries import IrregularTimeSeries, irregular_timeseries_no_padding_variable_time_collate_fn


parser = argparse.ArgumentParser('SEDformer Training')

parser.add_argument('--model', type=str, default='SEDformer', help="Model name")
parser.add_argument('--dataset', type=str, default='Wiki_article_missing_50pct', help="Dataset name")
parser.add_argument('--epoch', type=int, default=200, help="training epochs")
parser.add_argument('--patience', type=int, default=50, help="patience for early stop")
parser.add_argument('--history', type=int, default=90, help="historical window size")
parser.add_argument('--pred_len', type=int, default=30, help="prediction length")

parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")

parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
parser.add_argument('--align', type=str, default='no_padding', help='alignment method')
parser.add_argument('--split_ratio', type=str, default='6:2:2', help='train:val:test split ratio')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--data_path', type=str, default='data/', help="Data path")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_config():
    return {
        'input_dim': 1,
        'enc_channels_per_var': 8,
        'model_dim': int(os.environ.get('SED_DIM', 32)),
        'te_dim': 32,
        'spike_temp': 5.0,
        'lif_tau': float(os.environ.get('SED_LIF_TAU', 3.0)),
        'detach_reset': True,
        'trans_layers': int(os.environ.get('SED_LAYERS', 2)),
        'trans_heads': 4,
        'rff_features_num': 48,
        'pool_stride': int(os.environ.get('SED_POOL', 4)),
        'detach_feats': True
    }

def load_data():
    dataset = IrregularTimeSeries('data', args.dataset, device=device)
    
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(':'))
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    return train_dataset, val_dataset, test_dataset

def collate_fn(batch):
    return irregular_timeseries_no_padding_variable_time_collate_fn(
        batch, args, device, data_type="train"
    )

def create_data_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader

def compute_loss(predictions, targets, mask):
    mse_loss = nn.MSELoss(reduction='none')
    mae_loss = nn.L1Loss(reduction='none')
    
    mse = mse_loss(predictions, targets) * mask
    mae = mae_loss(predictions, targets) * mask
    
    mse = mse.sum() / mask.sum()
    mae = mae.sum() / mask.sum()
    
    return mse, mae

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # batch is a list of dictionaries
        batch_mse = 0.0
        batch_mae = 0.0
        batch_count = 0
        
        for sample in batch:
            observed_data = sample['observed_data'].to(device)
            observed_tp = sample['observed_tp'].to(device)
            observed_mask = sample['observed_mask'].to(device)
            data_to_predict = sample['data_to_predict'].to(device)
            tp_to_predict = sample['tp_to_predict'].to(device)
            mask_predicted_data = sample['mask_predicted_data'].to(device)
            
            predictions = model.forward_irregular([{
                'observed_data': observed_data,
                'observed_tp': observed_tp,
                'observed_mask': observed_mask,
                'tp_to_predict': tp_to_predict
            }])
            
            pred_tensor = predictions[0]
            
            mse_loss, mae_loss = compute_loss(pred_tensor, data_to_predict, mask_predicted_data)
            
            mse_loss.backward()
            
            batch_mse += mse_loss.item()
            batch_mae += mae_loss.item()
            batch_count += 1
        
        optimizer.step()
        
        total_mse += batch_mse / batch_count
        total_mae += batch_mae / batch_count
        num_batches += 1
    
    return total_mse / num_batches, total_mae / num_batches

def validate(model, val_loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # batch is a list of dictionaries
            batch_mse = 0.0
            batch_mae = 0.0
            batch_count = 0
            
            for sample in batch:
                observed_data = sample['observed_data'].to(device)
                observed_tp = sample['observed_tp'].to(device)
                observed_mask = sample['observed_mask'].to(device)
                data_to_predict = sample['data_to_predict'].to(device)
                tp_to_predict = sample['tp_to_predict'].to(device)
                mask_predicted_data = sample['mask_predicted_data'].to(device)
                
                predictions = model.forward_irregular([{
                    'observed_data': observed_data,
                    'observed_tp': observed_tp,
                    'observed_mask': observed_mask,
                    'tp_to_predict': tp_to_predict
                }])
                
                pred_tensor = predictions[0]
                
                mse_loss, mae_loss = compute_loss(pred_tensor, data_to_predict, mask_predicted_data)
                
                batch_mse += mse_loss.item()
                batch_mae += mae_loss.item()
                batch_count += 1
            
            total_mse += batch_mse / batch_count
            total_mae += batch_mae / batch_count
            num_batches += 1
    
    return total_mse / num_batches, total_mae / num_batches

def train():
    set_seed(args.seed)
    
    print(f"Training SEDformer on {args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epoch}")
    
    train_dataset, val_dataset, test_dataset = load_data()
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)
    
    model_config = get_model_config()
    model = SEDformer(**model_config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    
    best_val_mse = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(args.epoch):
        train_mse, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epoch}: "
              f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, "
              f"Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.save}/best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Load best model if it exists, otherwise use current model
    best_model_path = f"{args.save}/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No saved model found, using current model for testing")
    
    test_mse, test_mae = validate(model, test_loader, device)
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation MSE: {best_val_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    return test_mse, test_mae

if __name__ == '__main__':
    os.makedirs(args.save, exist_ok=True)
    train()
