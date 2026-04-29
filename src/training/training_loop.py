import torch

from src.training.engine import evaluate, train_one_epoch
from src.utils.common_functions import get_exp_dir


def training_loop(model,data,optimizer,criterion, config, logger,train_loader=None,val_loader=None):
    exp_dir = get_exp_dir(config['dataset'], config['model_type'])
    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    loss = 0
    for epoch in range(1, config['hparams']['epochs'] + 1):
        if train_loader == None:
            loss = train_one_epoch(model, data=data, optimizer=optimizer, criterion=criterion)
            val_metrics = evaluate(model, data = data, mask=data.val_mask)
            history['train_loss'].append(loss)
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_macro'])

        else:
            loss = train_one_epoch(model, loader=train_loader, optimizer=optimizer, criterion=criterion)
            val_metrics = evaluate(model, loader=val_loader)
            history['train_loss'].append(loss)
            history['val_f1'].append(val_metrics['f1'])

        if train_loader == None:
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                # torch.save(model.state_dict(), f'{model_save_path}/best_model.pt')
                torch.save(model.state_dict(), exp_dir / "best_model.pt")
                logger.info(f"Epoch {epoch:03d} | New Best Val Acc: {best_val_acc:.4f} - Saved!")

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        else:
            if val_metrics['f1'] > best_val_acc:
                best_val_acc = val_metrics['f1']
                torch.save(model.state_dict(), exp_dir / "best_model.pt")
                logger.info(f"Epoch {epoch:03d} | New Best Val F1: {best_val_acc:.4f} - Saved!")

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f}")
    return history