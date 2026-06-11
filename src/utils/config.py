# Inside src/utils/config.py

class Config:
    # Default (Cora)
    dataset_name = 'Cora'
    hidden_channels = 16
    num_layers = 2
    dropout = 0.5
    epochs = 200
    lr = 0.01
    weight_decay = 5e-4
    patience = 20
    device = 'cuda'
    seed = 42

    @classmethod
    def for_dataset(cls, name):
        # Override for specific datasets
        if name == 'PubMed':
            cls.hidden_channels = 64
            cls.epochs = 300
            cls.lr = 0.005
            cls.patience = 50
        elif name == 'CiteSeer':
            cls.patience = 30   # maybe
        return cls