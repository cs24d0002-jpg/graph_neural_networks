import sys
import torch
import platform
from src.data.data_loader import load_data
from src.models.model_factory import get_model
from src.training.engine import train_one_epoch, evaluate
from src.utils.config import load_config
from src.utils.common_functions import plot_training_results,get_exp_dir, setup_terminal_logger

def main():
    # 1. Setup
    config = load_config('configs/baseline_test.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data
    dataset = load_data(config['dataset'])
    data = dataset[0].to(device)
    
    # 3. Model
    model = get_model(
        config['model_type'], 
        dataset.num_features, 
        dataset.num_classes, 
        config['hparams']
    ).to(device)
    
    # 4. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hparams']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    exp_dir = get_exp_dir(config['dataset'], config['model_type'])
    logger = setup_terminal_logger(exp_dir)
    logger.info(f"Starting Experiment: {config['experiment_name']}")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Using Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(1, config['hparams']['epochs'] + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        val_metrics = evaluate(model, data, data.val_mask)
        
        # Record history
        history['train_loss'].append(loss)
        history['val_acc'].append(val_metrics['accuracy'])

        # Save Best Model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # torch.save(model.state_dict(), f'{model_save_path}/best_model.pt')
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            logger.info(f"Epoch {epoch:03d} | New Best Val Acc: {best_val_acc:.4f} - Saved!")

        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

    # Plot results after training
    plot_training_results(exp_dir,history)

if __name__ == "__main__":
    main()