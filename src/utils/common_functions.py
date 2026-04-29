import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
import sys
from src.utils.config import load_config
import torch
import platform

def initiate_logging():
    config = load_config('configs/baseline_test.yaml')
    exp_dir = get_exp_dir(config['dataset'], config['model_type'])
    logger = setup_terminal_logger(exp_dir)
    logger.info(f"Starting Experiment: {config['experiment_name']}")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Using Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Hyperparameters: {config['hparams']}")
    
    return logger

def plot_training_results(exp_dir, history):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss Plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history['train_loss'], color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Accuracy Plot
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history['val_acc'], color='tab:blue', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()
    plt.savefig(f'{exp_dir}/training_curve.eps', format='eps')
    # plt.show()
    
def get_exp_dir(dataset_name, model_name):
    # Generate timestamp: YYYYMMDD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Construct path: results/experiments/Cora/GCN/20260428_0930
    exp_path = Path("results/experiments") / dataset_name / model_name / timestamp
    
    # Create directory (and parents) if it doesn't exist
    exp_path.mkdir(parents=True, exist_ok=True)
    
    return exp_path


def setup_terminal_logger(exp_dir):
    """
    Redirects terminal output to both the console and a log file.
    """
    log_file = exp_dir / "terminal_output.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter for timestamps and messages
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # File Handler (Saves to disk)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler (Prints to terminal)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger