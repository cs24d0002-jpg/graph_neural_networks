import torch
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(logits, labels, mask, is_multi_label=False):
    """
    Computes performance metrics for GNN models.
    
    Args:
        logits: Raw output from the model
        labels: Ground truth labels
        mask: Boolean mask (train/val/test)
        is_multi_label: Set to True for datasets like PPI
    """
    # Filter data based on the mask
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    if is_multi_label:
        # For PPI: apply sigmoid and threshold at 0.5
        preds = (torch.sigmoid(masked_logits) > 0.5).cpu().numpy()
        labels_np = masked_labels.cpu().numpy()
        
        # Micro-F1 is the standard for PPI
        micro_f1 = f1_score(labels_np, preds, average='micro')
        return {"f1": micro_f1}
    
    else:
        # For Cora, CiteSeer, PubMed: use argmax
        preds = masked_logits.argmax(dim=1).cpu().numpy()
        labels_np = masked_labels.cpu().numpy()
        
        acc = accuracy_score(labels_np, preds)
        # Macro-F1 is helpful for imbalanced datasets like PubMed
        macro_f1 = f1_score(labels_np, preds, average='macro')
        
        return {
            "accuracy": acc,
            "f1_macro": macro_f1
        }