import math
import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Union
import lightning as L
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from rich.console import Console
from torch import Tensor
from torchmetrics import MatthewsCorrCoef, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score, SpearmanCorrCoef
from torchmetrics.regression import MeanAbsolutePercentageError
import wandb
from aigpro.utils.logger import get_logger

console = Console()
logger = get_logger()

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3) -> Callable[..., float]:
    def scaler(x):
        return 1.0

    def lr_lambda(it):
        return min_lr + (max_lr - min_lr) * relative(it, stepsize)

    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

class PDBModelModule(L.LightningModule):
    def __init__(
        self,
        model=None,
        learning_rate: float = 1e-4,
        optimizer_name: str = "Adam",
        batch_size: int = 32,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        num_workers: Union[int, None] = None,
        weight_decay=1e-2,
        scheduler_name: str = "ReduceLROnPlateau",
        scheduler_monitor: str = "loss",
        decay_milestone: Union[None, List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        decay_milestone = decay_milestone or [50, 80, 110, 150, 200, 220, 250]
        self.multi = False
        self.scheduler_monitor: str = scheduler_monitor
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.num_workers: Union[int, None] = num_workers or os.cpu_count()
        self.model = model
        self.mse: MeanSquaredError = MeanSquaredError()
        self.rmse: MeanSquaredError = MeanSquaredError(squared=False)
        self.r2: R2Score = R2Score()
        self.mae: MeanAbsoluteError = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.spearmanr: SpearmanCorrCoef = SpearmanCorrCoef()
        self.pearsonr = PearsonCorrCoef()
        self.matthews_corrcoef = MatthewsCorrCoef(task="binary")
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        self.result_dict_train = {}
        self.result_dict_test = {}
        self.result_dict_valid = {}
        self.schedulers_name = scheduler_name
        self.alpha_custom = 0.7
        self.validation_step_outputs = []
        self.test_step_outputs = {}
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = True

    def configure_optimizers(self) -> Tuple[List, List]:
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name == "Adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        elif self.hparams.optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        if self.hparams.scheduler_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.decay_milestone, gamma=0.5)
        elif self.hparams.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
        elif self.hparams.scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.8, patience=3, verbose=True, threshold=0.0001, threshold_mode="rel", cooldown=0, min_lr=1e-12, eps=1e-08
            )
        elif self.hparams.scheduler_name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 5,
            },
        }

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.accumulate_grad_batches * self.epochs

    def forward(self, x) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> dict:
        loss, y_true, y_pred = self.compute_loss(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs: dict[str, Tensor] = {"train_rmse_loss": loss}
        progress_bar_metrics: dict[str, Tensor] = tensorboard_logs
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def compute_loss(self, batch):
        x, y_true = batch
        y_true, y_label = y_true
        y_pred_class, y_pred = self.all_prediction(x)
        loss = self.mse(y_pred, y_true)
        return loss, y_true, y_pred

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss, y_true, y_pred = self.compute_loss(batch)
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "pred": y_pred,
                "true": y_true,
            }
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def on_validation_epoch_end(self) -> None:
        _y_pred, _y_true, results = self.metric_and_log(self.validation_step_outputs, title="val", log_plot=True)
        print_metric_table("Validation Metrics", asdict(results))
        self.validation_step_outputs.clear()
        del results

    def all_prediction(self, x):
        result = self.model(x)
        result = result.flatten()
        return result

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx) -> Dict[str, Tensor]:
        if dataloader_idx > 0:
            self.multi = True
        loss, y_true, y_pred = self.compute_loss(batch)
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = {
                "val_loss": [],
                "pred": [],
                "true": [],
            }
        self.test_step_outputs[dataloader_idx]["val_loss"].append(loss)
        self.test_step_outputs[dataloader_idx]["pred"].append(y_pred)
        self.test_step_outputs[dataloader_idx]["true"].append(y_true)
        self.log(f"test_loss_{dataloader_idx}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            f"loss_{dataloader_idx }": loss,
            f"pred_{dataloader_idx }": y_pred,
            f"true_{dataloader_idx }": y_true,
        }

    def on_test_epoch_end(self) -> None:
        if self.multi:
            _y_pred, _y_true, results = self.multi_compute_metrics(self.test_step_outputs)
            for i in range(len(results)):
                print_metric_table(f"Test Metrics {i}", asdict(results[i]))
        else:
            _y_pred, _y_true, results = self.metric_and_log(self.test_step_outputs, title="test", log_plot=True)
            print_metric_table("Test Metrics", asdict(results))
        self.test_step_outputs.clear()

    def get_backbone(self):
        return self.model

    def compute_metrics(self, y_pred, y_true):
        y_pred = y_pred.to(torch.float32)
        y_true = y_true.to(torch.float32)
        metric_mse = self.mse(y_pred, y_true)
        metric_mae = self.mae(y_pred, y_true)
        metric_rmse = self.rmse(y_pred, y_true)
        metric_r2 = self.r2(y_pred, y_true)
        metric_spear = self.spearmanr(y_pred, y_true)
        metric_pearsonr = self.pearsonr(y_pred, y_true)
        concordnace_index = concordance_index(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return (
            metric_mse,
            metric_mae,
            metric_rmse,
            metric_r2,
            metric_spear,
            metric_pearsonr,
            concordnace_index,
        )

    def multi_compute_metrics(self, test_step_outputs: dict):
        all_y_pred, y_true, results = [], [], []
        for k, v in test_step_outputs.items():
            outputs = []
            y_pred = []
            y_true = []
            y_pred.append(v["pred"])
            y_true.append(v["true"])
            for i in range(len(v["pred"])):
                outputs.append({"pred": v["pred"][i], "true": v["true"][i]})
            _y, _t, _r = self.metric_and_log(outputs, title=f"test_{k}", log_plot=True)
            all_y_pred.append(_y)
            y_true.append(_t)
            results.append(_r)
        return all_y_pred, y_true, results

    def metric_and_log(self, outputs, title, log_plot=True):
        y_pred, y_true = zip(*[(x["pred"], x["true"]) for x in outputs if all(k in x for k in ("pred", "true"))])
        y_pred = torch.cat(y_pred, dim=0).detach()
        y_true = torch.cat(y_true, dim=0).detach()
        (
            _mse,
            _mae,
            _rmse,
            _r2,
            _spear,
            _pearsonr,
            _ci,
        ) = self.compute_metrics(y_pred, y_true)
        results = Metrics(
            mse=_mse,
            rmse=_rmse,
            mae=_mae,
            r2=_r2,
            spearman=_spear,
            pearson=_pearsonr,
            ci=_ci,
        )
        self.log_dict(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        wandb.log(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            }
        )
        if log_plot:
            fig = scatter_plot(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                title=f"{title} scatter plot",
            )
            wandb.log({f"{title}_scatter_plot": wandb.Image(fig)})
            fig.clf()
        return y_pred, y_true, results

import lightning as L
import torch
import torch.nn as nn
from aigpro.models.models import BestGPCR
from torchmetrics.classification import Accuracy

class PDBModelModuleGPCR(L.LightningModule):
    def __init__(self, learning_rate=1e-4, batch_size=32):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = BestGPCR()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.automatic_optimization = True
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_label = y
        y_label = y_label.to(torch.long)
        logits = self.model(x)
        loss = self.loss_fn(logits, y_label)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y_label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_label = y
        y_label = y_label.to(torch.long)
        logits = self.model(x)
        loss = self.loss_fn(logits, y_label)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y_label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            avg_train_loss = sum(self.train_losses[-len(train_dataloader):]) / len(train_dataloader)
            print(f"Epoch {self.current_epoch} - Average Train Loss: {avg_train_loss:.4f}")
        else:
            avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0
            print(f"Epoch {self.current_epoch} - Average Train Loss: {avg_train_loss:.4f}")

        if self.val_losses:
            avg_val_loss = sum(self.val_losses[-len(self.trainer.datamodule.val_dataloader()):]) / len(self.trainer.datamodule.val_dataloader())
            print(f"Epoch {self.current_epoch} - Average Val Loss: {avg_val_loss:.4f}")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # batch 전체를 GPU로 이동
        batch = [item.to('cuda:0') if isinstance(item, torch.Tensor) else item for item in batch]
        x = batch[0]  # x는 이미 GPU에 있는 상태로 추출
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return probs