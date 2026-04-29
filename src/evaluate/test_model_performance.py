from sklearn.metrics import f1_score
import torch
from src.evaluate.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model,exp_dir, data, config, logger,is_loader = False):
    if is_loader == False:
        model.load_state_dict(torch.load(exp_dir / "best_model.pt",weights_only=True))
        model.eval()
        with torch.no_grad():
            # Forward pass through the whole graph
            out = model(data.x, data.edge_index)
            
            # Calculate metrics on unseen labels
            test_metrics = compute_metrics(out, data.y, data.test_mask, is_multi_label=(config['dataset'] == 'PPI'))

        logger.info(f"Unseen Test Accuracy: {test_metrics['accuracy']:.4f} | F1 Score: {test_metrics['f1_macro']:.4f}")
    else:
        model.load_state_dict(torch.load(exp_dir / "best_model.pt",weights_only=True))
        model.eval()
        all_preds, all_targets = [], []
        for batch in data:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            # Threshold at 0.5 for multi-label prediction
            pred = (logits > 0).float() 
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.cpu())
        
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        logger.info(f"Unseen Test F1 Score: {f1_score(y_true, y_pred, average='micro'):.4f}")