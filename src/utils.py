import os
import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, forces deterministic CUDA ops and disables
            cuDNN auto-tuning. Slightly slower but fully reproducible.
            Set to False only for production throughput benchmarking.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUBLAS workspace config required for torch.use_deterministic_algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False       # disable auto-tuner
        torch.backends.cudnn.deterministic = True    # force deterministic kernels
        # warn_only=True: logs warnings instead of raising errors for ops
        # that lack a deterministic implementation (e.g., scatter on older GPUs).
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Speed-optimised: non-deterministic but faster
        torch.backends.cudnn.benchmark = True

def get_worker_init_fn(base_seed: int):
    """Return a DataLoader worker_init_fn that seeds each worker process.

    Each worker receives a unique seed derived from base_seed and its worker_id,
    preventing correlated augmentation/sampling across workers.

    Usage:
        loader = DataLoader(...,
                            worker_init_fn=get_worker_init_fn(args.seed),
                            generator=torch.Generator().manual_seed(args.seed))
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn

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
