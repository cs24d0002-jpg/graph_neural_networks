#packages
import torch
import argparse

#modular code import
from src.evaluate.test_model_performance import test_model
from src.data.data_loader import load_data
from src.models.model_factory import get_model
from src.utils.config import load_config
from src.utils.common_functions import initiate_logging, plot_training_results,get_exp_dir
from src.training.training_loop import training_loop

def parse_args():
    parser = argparse.ArgumentParser(description="GNN Research Baseline Runner")
    
    # Experiment Setup
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora, CiteSeer, PubMed, PPI, Reddit')
    parser.add_argument('--model_type', type=str, default='gcn', help='gcn, gat')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for PPI/Reddit')
    
    args = parser.parse_args()
    return args


def main():

    config = load_config('configs/baseline_test.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = initiate_logging()
    exp_dir = get_exp_dir(config['dataset'], config['model_type'])
    
    
    if(config['dataset'].lower() in ['cora', 'citeseer', 'pubmed']):
        
        logger.info(f"Loading {config['dataset']} dataset...")
        dataset = load_data(config['dataset'])
        data = dataset[0].to(device)
        data.test_mask = data.test_mask if hasattr(data, 'test_mask') else None

        criterion = torch.nn.CrossEntropyLoss()

        model = get_model(
            config['model_type'], 
            dataset.num_features, 
            dataset.num_classes, 
            config['hparams']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['hparams']['lr'])
        history = training_loop(model, data, optimizer, criterion, config, logger)
    
        test_model(model, exp_dir, data, config, logger)
        plot_training_results(exp_dir,history)
    elif(config['dataset'].lower() in ['ppi']):
        logger.info(f"Loading {config['dataset']} dataset...")
        train_loader, val_loader, test_loader, in_channels, out_channels = load_data(config['dataset'])
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        model = get_model(
            config['model_type'], 
            in_channels, 
            out_channels, 
            config['hparams']
        ).to(device)
        
        
        
        
if __name__ == "__main__":
    main()