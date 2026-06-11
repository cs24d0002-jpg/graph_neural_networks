import sys
import os
import time
import torch
import torch.optim as optim
from src.data_loader.cora_loader import load_cora
from src.models.gcn import GCN
from src.training.trainer import train_full_batch
from src.training.early_stop import EarlyStopping
from src.evaluate.metrics import evaluate_full_batch
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.seed import set_seed

def main():
    # --- Setup ---
    config = Config()
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Optional: log to file
    if not os.path.exists('logs'):
        os.makedirs('logs')
    sys.stdout = Logger(log_file=f'logs/cora_gcn_{config.seed}.log')
    
    # --- Load data ---
    print('Loading Cora dataset...')
    dataset, data = load_cora()          # <-- FIXED: get dataset and data
    data = data.to(device)
    
    # --- Model, optimizer, early stopping ---
    model = GCN(
        in_channels=dataset.num_features,      # <-- from dataset
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,      # <-- from dataset
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    early_stopping = EarlyStopping(patience=config.patience)
    
    # --- Training loop with epoch timing ---
    print('Start training...')
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        
        loss = train_full_batch(model, data, optimizer, data.train_mask, device)
        
        # Evaluate on validation set
        val_acc, val_f1 = evaluate_full_batch(model, data, data.val_mask, device)
        test_acc, test_f1 = evaluate_full_batch(model, data, data.test_mask, device)
        
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        
        # Save best models
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
        # Early stopping
        early_stopping(val_acc)
        
        # Print with epoch time
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Time: {epoch_time:.2f}s | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}')
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # --- Final evaluation on best model ---
    print('\n--- Final Results ---')
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    final_test_acc, final_test_f1 = evaluate_full_batch(model, data, data.test_mask, device)
    print(f'Test Accuracy: {final_test_acc:.4f}')
    print(f'Test Micro-F1:  {final_test_f1:.4f}')

if __name__ == '__main__':
    main()