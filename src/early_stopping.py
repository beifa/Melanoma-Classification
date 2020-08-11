import torch
import numpy as np 
import os

class EarlyStopping:
    def __init__(self, step = 5, mode = 'max'):
        """
        step : how long not change score
        mode : up/lo
        
        """
        self.step = step
        self.mode = mode
        self.count = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = [-np.Inf if mode == 'min' else np.Inf][0]
        

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score:
            self.count += 1
            
            if self.count >= self.step:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.count = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        name = type(model).__name__
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            torch.save(model.state_dict(), os.path.join(model_path, f'checkpoint_early_stop_{name}.pth'))
        self.val_score = epoch_score