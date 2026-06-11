import sys
import os
import time
import torch
import torch.optim as optim

from src.data_loader.reddit_loader import load_reddit
from src.models.gcn import GCN
from src.training.trainer_reddit import train_reddit
from src.evaluate.metrics_reddit import evaluate_reddit
from src.utils.config_reddit import ConfigReddit
from src.utils.logger import Logger
from src.utils.seed import set_seed

def main():
    config = ConfigReddit()
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists('logs'):
        os.makedirs('logs')
    sys.stdout = Logger(log_file=f'logs/reddit_gcn_{config.seed}.log')

    print('Loading Reddit dataset...')
    dataset, data, train_loader, subgraph_loader = load_reddit()

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print('Start training...')
    best_val_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_reddit(model, train_loader, optimizer, device)
        val_acc = evaluate_reddit(model, data, subgraph_loader, data.val_mask, device)

        epoch_time = time.perf_counter() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_reddit.pt')

        print(f'Epoch {epoch:03d} | Time: {epoch_time:.2f}s | Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

    print('\n--- Final Results ---')
    model.load_state_dict(torch.load('best_model_reddit.pt', weights_only=True))
    test_acc = evaluate_reddit(model, data, subgraph_loader, data.test_mask, device)
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()