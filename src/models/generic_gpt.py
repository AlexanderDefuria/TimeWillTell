import os
from pathlib import Path
from typing import Any, Callable
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from src.datasets.dataset import GenericDataset
from src.models import get_hf_cache_dir


def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    return bnb_config


def get_loss_fn(loss_fn: str) -> Callable:
    if loss_fn == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == "ce":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")


SAVED_HPARAMS = [
    "model_name",
    "lr",
    "num_labels",
    "loss_fn",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "dataset_name",
    "batch_size",
    "max_length",
    "instruction",
    "seed",
    "tokenizer",
    "accumulate_grad_batches",
    "workers",
    "dataset_size",
    "weight_decay",
]


class GenericGPTModel(pl.LightningModule):
    """LoRA fine-tuned model for source code vulnerability detection (binary classification)."""

    def __init__(
        self,
        model_name: str,
        lr: float,
        num_labels: int,
        loss_fn: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] = [],
        scheduler: str = "constant",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super(GenericGPTModel, self).__init__()

        # Save all the possible hyperparameters that have been passed in
        saved_hparams = {k: v for k, v in locals().items() if k in SAVED_HPARAMS}
        saved_kwargs = {k: v for k, v in kwargs.items() if k in SAVED_HPARAMS}
        saved_hparams = {**saved_hparams, **saved_kwargs}
        print(f"Saved hyperparameters: {saved_hparams}")
        self.save_hyperparameters(saved_hparams)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            quantization_config=get_bnb_config(),
            torch_dtype=torch.bfloat16,
            cache_dir=get_hf_cache_dir(),
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
        )
        self.tokenizer = GenericDataset.get_tokenizer(model_name)
        self.weight_decay = weight_decay
        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore
        self.model.config.pad_token_id = self.tokenizer.pad_token_id  # type: ignore
        self.lora_config = LoraConfig(
            r=lora_r,  # The rank of the update matrices. Lower rank means fewer parameters to train.
            lora_alpha=lora_alpha,  # A scaling factor for the LoRA weights.
            target_modules=target_modules,  # List of module names to apply LoRA to.
            lora_dropout=lora_dropout,  # Dropout probability for LoRA layers.
            bias="none",
            task_type=TaskType.SEQ_CLS,  # Task type for sequence classification
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.holdout_test: bool = False
        self.loss_fn = get_loss_fn(loss_fn)
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.lr = lr
        self.args = kwargs
        self.batch_size = kwargs.get("batch_size", 8)
        self.max_length = kwargs.get("max_length", 1024)
        self.scheduler = scheduler
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.validation_step_indices = []
        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_step_indices = []
        self.training_step_outputs = []
        self.training_step_labels = []
        self.training_step_indices = []

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask.squeeze(1),
        )
        logits = outputs.logits
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]

    def logging(self, label, metric, batch_size):
        self.log(label, metric, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def save_predictions_and_labels(self, predictions: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, phase: str) -> None:
        # Save predictions and labels to a file for later analysis
        if self.holdout_test:
            phase = "holdout"
            self.holdout_test = False

        # Get the log directory
        assert self.logger is not None, "Logger is not initialized."
        assert self.logger.log_dir is not None, "Logger log_dir is not set."
        log_dir = Path(self.logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(getattr(self, "current_epoch", 0))
        step = int(getattr(self, "global_step", 0))
        results_file = log_dir / f"{phase}_predictions_labels_epoch{epoch:03d}_step{step:08d}.parquet"
        df = pd.DataFrame(
            {
                "index": indices.cpu().numpy(),
                "prediction": predictions.cpu().numpy(),
                "label": labels.cpu().numpy(),
            }
        )
        df.to_parquet(results_file, index=False)

    def state_dict(self, **kwargs) -> dict[str, Any]:  # type: ignore
        """
        Override state_dict to save the PEFT model state dict.
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:  # type: ignore
        """
        rename from base_model to model.base_model
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_model."):
                new_k = f"model.{k}"
            else:
                new_k = k
            new_state_dict[new_k] = v
        super().load_state_dict(new_state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        return super().on_load_checkpoint(checkpoint)

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Common forward pass for training, validation, and test steps.

        Args:
            batch: A batch dict with input_ids, attention_mask, label, index

        Returns:
            (loss, preds, labels, indices, batch_size)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        indices = batch["index"]
        labels = F.one_hot(batch["label"].to(torch.int64), num_classes=2).to(dtype=torch.bfloat16)
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        preds = outputs.argmax(dim=-1).reshape(-1)
        labels = labels.argmax(dim=-1).reshape(-1)
        batch_size = input_ids.shape[0]
        return loss, preds, labels, indices, batch_size

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices, batch_size = self._shared_step(batch)
        self.training_step_outputs.append(preds)
        self.training_step_labels.append(labels)
        self.training_step_indices.append(indices)
        self.logging("train_loss", loss, batch_size=batch_size)
        return loss

    def _log_classification_metrics(self, phase: str, preds: torch.Tensor, labels: torch.Tensor, sync_dist: bool = True) -> None:
        self.log(f"{phase}_f1", self.f1(preds, labels), prog_bar=True, sync_dist=sync_dist)
        self.log(f"{phase}_accuracy", (preds == labels).float().mean(), prog_bar=True, sync_dist=sync_dist)
        self.log(f"{phase}_recall", self.recall(preds, labels), prog_bar=False, sync_dist=sync_dist)
        self.log(f"{phase}_precision", self.precision(preds, labels), prog_bar=False, sync_dist=sync_dist)
        self.log(f"{phase}_balance", labels.float().mean(), prog_bar=False, sync_dist=sync_dist)

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) == 0 or len(self.training_step_labels) == 0:
            return
        all_preds = torch.cat(self.training_step_outputs)
        all_labels = torch.cat(self.training_step_labels)
        all_indices = torch.cat(self.training_step_indices)
        self.save_predictions_and_labels(all_preds, all_labels, all_indices, phase="train")
        self._log_classification_metrics("train", all_preds, all_labels)
        self.training_step_outputs.clear()
        self.training_step_labels.clear()
        self.training_step_indices.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices, batch_size = self._shared_step(batch)
        self.validation_step_outputs.append(preds)
        self.validation_step_labels.append(labels)
        self.validation_step_indices.append(indices)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0 or len(self.validation_step_labels) == 0:
            return
        all_preds = torch.cat(self.validation_step_outputs)
        all_labels = torch.cat(self.validation_step_labels)
        all_indices = torch.cat(self.validation_step_indices)
        self.save_predictions_and_labels(all_preds, all_labels, all_indices, phase="val")
        if self.trainer.global_step > 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad.float(), self.trainer.global_step)
                if param.data is not None:
                    self.logger.experiment.add_histogram(f"{name}_weight", param.data.float(), self.trainer.global_step)
        self._log_classification_metrics("val", all_preds, all_labels)
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()
        self.validation_step_indices.clear()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch = args[0]
        loss, preds, labels, indices, batch_size = self._shared_step(batch)
        self.logging("test_loss", loss, batch_size=batch_size)
        self.test_step_outputs.append(preds)
        self.test_step_labels.append(labels)
        self.test_step_indices.append(indices)
        return loss

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0 or len(self.test_step_labels) == 0:
            return
        all_preds = torch.cat(self.test_step_outputs)
        all_labels = torch.cat(self.test_step_labels)
        all_indices = torch.cat(self.test_step_indices)
        self.save_predictions_and_labels(all_preds, all_labels, all_indices, phase="test")
        self._log_classification_metrics("test", all_preds, all_labels, sync_dist=False)
        self.log("test_size", len(all_labels), prog_bar=True)
        self.test_step_outputs.clear()
        self.test_step_labels.clear()
        self.test_step_indices.clear()
