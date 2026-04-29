#packages
import torch
import argparse

#modular code import
from src.evaluate.test_model_performance import test_model
from src.data.data_loader import load_data
from src.models.model_factory import get_model
from src.utils.common_functions import initiate_logging, plot_training_results,get_exp_dir,load_config
from src.training.training_loop import training_loop

print("Hello! This is the main file for running GNN research baselines. You can specify the dataset and model type in the configuration file (e.g., configs/baseline_test.yaml). The code is modular, with separate files for data loading, model definition, training, and evaluation. Make sure to have the necessary datasets downloaded and placed in the correct directories as expected by PyG.")
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

    print("Main File ----Configurations and Setup")
    config = load_config('configs/baseline_test.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = initiate_logging()
    exp_dir = get_exp_dir(config['dataset'], config['model_type'])
    
    dataset = None
    train_loader, val_loader, test_loader = None, None, None
    
    #Dataset
    logger.info(f"Loading {config['dataset']} dataset...")
    if(config['dataset'].lower() in ['cora', 'citeseer', 'pubmed']):
        dataset = load_data(config['dataset'])
        data = dataset[0].to(device)
        data.test_mask = data.test_mask if hasattr(data, 'test_mask') else None
    else:
        train_loader, val_loader, test_loader, in_channels, out_channels = load_data(config['dataset'])
    criterion = torch.nn.CrossEntropyLoss()
        
    #Loss Function
    if(config['dataset'].lower() =='ppi'):
        criterion = torch.nn.BCEWithLogitsLoss()

    #Model Definition
    model = get_model(
        config['model_type'], 
        dataset.num_features, 
        dataset.num_classes, 
        config['hparams']
    ).to(device)
    logger.info("--- Model Architecture ---")
    logger.info(f"\n{str(model)}") # This logs the layers and parameters
    
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hparams']['lr'])
    
    if(dataset is not None):
        #Model Training and Validation
        history = training_loop(model=model, data=data, optimizer=optimizer, criterion=criterion, config=config, logger=logger)

        #Model Testing
        test_model(model, exp_dir, data, config, logger)
    else:
        #Model Training and Validation
        history = training_loop(model=model, data=None, optimizer=optimizer, criterion=criterion, config=config, logger=logger, train_loader=train_loader,val_loader=val_loader)

        #Model Testing
        test_model(model, exp_dir, test_loader, config, logger,is_loader = True)
    
    #Plotting Training Results
    plot_training_results(exp_dir,history)
if __name__ == "__main__":
    main()