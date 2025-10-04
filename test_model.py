#!/usr/bin/env python3

import torch
from model.SEDformer import SEDformer

def test_model():
    print("Testing SEDformer model loading...")
    
    # Create model with default parameters
    model = SEDformer(
        input_dim=1,
        enc_channels_per_var=8,
        model_dim=32,
        te_dim=32,
        spike_temp=5.0,
        lif_tau=3.0,
        detach_reset=True,
        trans_layers=2,
        trans_heads=4,
        rff_features_num=48,
        pool_stride=4,
        detach_feats=True
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 100
    pred_len = 30
    
    # Create dummy input data
    X = torch.randn(batch_size, seq_len, 1)
    tp_true = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    tp_pred = torch.arange(seq_len, seq_len + pred_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones_like(X)
    
    print(f"Input shapes:")
    print(f"  X: {X.shape}")
    print(f"  tp_true: {tp_true.shape}")
    print(f"  tp_pred: {tp_pred.shape}")
    print(f"  mask: {mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model.forecasting(tp_pred, X, tp_true, mask)
    
    print(f"Output shape: {output.shape}")
    print("✓ Model test passed!")

if __name__ == "__main__":
    test_model()
