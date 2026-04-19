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
            Set False only for production throughput benchmarking.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required when torch.use_deterministic_algorithms is enabled.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False       # disable fastest-kernel auto-search
        torch.backends.cudnn.deterministic = True    # force deterministic cuDNN kernels
        # warn_only=True: log warnings instead of raising errors for ops
        # that lack a deterministic CUDA kernel (e.g. scatter on older GPUs).
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Speed-optimised mode: non-deterministic but faster.
        torch.backends.cudnn.benchmark = True


def get_worker_init_fn(base_seed: int):
    """Return a DataLoader worker_init_fn that seeds each worker process.

    Each worker receives a unique but deterministic seed (base_seed + worker_id),
    preventing correlated sampling across workers while keeping runs reproducible.

    Usage:
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            worker_init_fn=get_worker_init_fn(args.seed),
            generator=g,
        )
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


class EarlyStopping:
    """Early stops training if validation score does not improve after patience epochs.

    Saves a full checkpoint including model, optimizer, scaler, epoch, args,
    and RNG states so that training can be resumed exactly from the best epoch.
    """

    def __init__(self, patience: int = 10, verbose: bool = False,
                 delta: float = 0.0, path: str = 'checkpoint.pt',
                 mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement before stopping.
            verbose: Print a message when validation score improves.
            delta: Minimum change to qualify as an improvement.
            path: Filepath to save the best checkpoint.
            mode: 'min' to minimise (e.g. loss) or 'max' to maximise (e.g. BACC).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, score: float, model: torch.nn.Module,
                 optimizer=None, scaler=None,
                 epoch: int = None, args=None) -> None:
        """Evaluate score and save checkpoint if improved.

        Args:
            score: Current validation metric value.
            model: Model whose state_dict will be saved.
            optimizer: Optional optimizer for resumable checkpoints.
            scaler: Optional AMP GradScaler for resumable checkpoints.
            epoch: Current epoch number (stored in checkpoint).
            args: Parsed argparse namespace (stored in checkpoint).
        """
        current_score = score if self.mode == 'max' else -score
        best_comp = self.val_score_best if self.mode == 'max' else -self.val_score_best

        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(score, model, optimizer, scaler, epoch, args)
        elif current_score < best_comp + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self._save_checkpoint(score, model, optimizer, scaler, epoch, args)
            self.counter = 0

    def _save_checkpoint(self, score: float, model: torch.nn.Module,
                         optimizer=None, scaler=None,
                         epoch: int = None, args=None) -> None:
        """Save a full training checkpoint to self.path."""
        if self.verbose:
            print(
                f'Validation score improved '
                f'({self.val_score_best:.6f} --> {score:.6f}). Saving model...'
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "val_score": score,
            # RNG states: allows exact bit-for-bit resume from this epoch.
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if args is not None:
            checkpoint["args"] = vars(args)

        torch.save(checkpoint, self.path)
        self.val_score_best = score
