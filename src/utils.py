import random
import numpy as np
import torch

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, score, model):
        if self.mode == 'min':
            current_score = -score
            best_score_comp = -self.val_score_best
        else: 
            current_score = score
            best_score_comp = self.val_score_best

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(score, model)
        elif current_score < best_score_comp + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_score_best:.6f} --> {score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_best = score