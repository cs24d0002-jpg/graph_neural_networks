import torch
from src.evaluate.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model,exp_dir, data, config, logger):
    
    model.load_state_dict(torch.load(exp_dir / "best_model.pt",weights_only=True))
    model.eval()
    with torch.no_grad():
        # Forward pass through the whole graph
        out = model(data.x, data.edge_index)
          
        # Calculate metrics on unseen labels
        test_metrics = compute_metrics(out, data.y, data.test_mask, is_multi_label=(config['dataset'] == 'PPI'))

    logger.info(f"Unseen Test Accuracy: {test_metrics['accuracy']:.4f} | F1 Score: {test_metrics['f1_macro']:.4f}")