from .gnn_baselines import GCN, GAT, GraphSAGE

def get_model(model_name, in_channels, out_channels, hparams):
    model_name = model_name.lower()
    
    if model_name == 'gcn':
        return GCN(in_channels, out_channels, 
                   hidden_channels=hparams.get('hidden_channels', 64),
                   dropout=hparams.get('dropout', 0.5))
    
    elif model_name == 'gat':
        return GAT(in_channels, out_channels, 
                   hidden_channels=hparams.get('hidden_channels', 8),
                   heads=hparams.get('heads', 8))
    
    elif model_name == 'sage':
        return GraphSAGE(in_channels, out_channels, 
                         hidden_channels=hparams.get('hidden_channels', 64))
    
    else:
        raise ValueError(f"Model {model_name} not defined in factory.")