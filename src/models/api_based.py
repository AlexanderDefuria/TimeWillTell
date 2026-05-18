from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from openai import OpenAI

from openai import BadRequestError

from src.models.prompts import ZERO_SHOT_PROMPT, SYS_INST

MAX_NEW_TOKENS = 32


class ContextExceededError(RuntimeError):
    """Raised when a request exceeds the model's available context window."""


class ApiBasedModel(pl.LightningModule):
    """Zero-shot model that queries an OpenAI-compatible API (e.g. llama-server) instead of loading weights locally."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        instruction: str,
        tokenizer: str,
        num_labels: int = 2,
        system_prompt: str = SYS_INST,
        user_prompt_template: str = ZERO_SHOT_PROMPT,
        seed: int = 42,
        extra_body_params: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name
        self.extra_body_params = extra_body_params or {}
        self.instruction = instruction
        self.num_labels = num_labels
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.seed = seed
        self.api_url = api_url.rstrip("/")
        self.client = OpenAI(base_url=f"{self.api_url}/v1", api_key="none")

        self.holdout_test: bool = False
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.test_step_outputs: list[torch.Tensor] = []
        self.test_step_labels: list[torch.Tensor] = []
        self.test_step_indices: list[torch.Tensor] = []
        self.test_step_answers: list[str] = []

        self.save_hyperparameters(
            {
                "api_url": api_url,
                "model_name": model_name,
                "instruction": instruction,
                "num_labels": num_labels,
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
            }
        )

    def configure_optimizers(self):
        return []

    def _log_classification_metrics(self, phase: str, preds: torch.Tensor, labels: torch.Tensor, sync_dist: bool = True) -> None:
        self.log(f"{phase}_f1", self.f1(preds, labels), prog_bar=True, sync_dist=sync_dist)
        self.log(f"{phase}_accuracy", (preds == labels).float().mean(), prog_bar=True, sync_dist=sync_dist)
        self.log(f"{phase}_recall", self.recall(preds, labels), prog_bar=False, sync_dist=sync_dist)
        self.log(f"{phase}_precision", self.precision(preds, labels), prog_bar=False, sync_dist=sync_dist)
        self.log(f"{phase}_balance", labels.float().mean(), prog_bar=False, sync_dist=sync_dist)

    def _generate_answer(self, code: str) -> str:
        return self._generate_answer_with_prompt(code, self.user_prompt_template)

    def _generate_answer_with_prompt(self, code: str, prompt_template: str) -> str:
        user_prompt = prompt_template.format(func=code)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_NEW_TOKENS,
                temperature=0,
                seed=self.seed,
                **({"extra_body": self.extra_body_params} if self.extra_body_params else {}),
            )
        except BadRequestError as exc:
            if "exceed" in str(exc).lower() and "context" in str(exc).lower():
                raise ContextExceededError(
                    f"Request exceeds context window (url={self.api_url}, model={self.model_name}): {exc}"
                ) from exc
            raise RuntimeError(
                f"API call failed (url={self.api_url}, model={self.model_name}): {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"API call failed (url={self.api_url}, model={self.model_name}): {exc}"
            ) from exc
        return response.choices[0].message.content.strip()

    @staticmethod
    def _to_binary_label(answer: str) -> int:
        normalized = answer.strip().upper()
        if normalized.startswith(("YES", "(1)")):
            return 1
        if normalized.startswith(("NO", "(2)")):
            return 0
        return 0

    def save_predictions_and_labels(
        self, predictions: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, phase: str
    ) -> None:
        if self.holdout_test:
            phase = "holdout"
            self.holdout_test = False

        assert self.logger is not None, "Logger is not initialized."
        assert self.logger.log_dir is not None, "Logger log_dir is not set."
        log_dir = Path(self.logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(getattr(self, "current_epoch", 0))
        step = int(getattr(self, "global_step", 0))
        results_file = log_dir / f"{phase}_predictions_labels_epoch{epoch:03d}_step{step:08d}.parquet"
        pd.DataFrame(
            {
                "index": indices.cpu().numpy(),
                "prediction": predictions.cpu().numpy(),
                "label": labels.cpu().numpy(),
                "answer": self.test_step_answers,
            }
        ).to_parquet(results_file, index=False)

    def on_test_epoch_end(self):
        if not self.test_step_outputs or not self.test_step_labels:
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
        self.test_step_answers.clear()

    def test_step(self, *args: Any, **kwargs: Any):
        batch = args[0]
        labels = batch["label"].to(torch.int64).reshape(-1)
        indices = batch["index"]
        code_texts = batch["text"]
        predictions = []
        answers = []
        skipped = 0
        for code in code_texts:
            try:
                answer = self._generate_answer(code)
            except ContextExceededError:
                skipped += 1
                if skipped == 1:
                    print("[test_step] Warning: sample exceeded context window, defaulting prediction to 0.")
                answers.append("")
                predictions.append(0)
                continue
            answers.append(answer)
            predictions.append(self._to_binary_label(answer))
        if skipped:
            print(f"[test_step] Skipped {skipped}/{len(code_texts)} samples (context exceeded), predicted 0.")

        pred_tensor = torch.tensor(predictions, dtype=torch.int64, device=labels.device)
        loss = (pred_tensor != labels).float().mean()
        batch_size = len(code_texts)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.test_step_outputs.append(pred_tensor)
        self.test_step_labels.append(labels)
        self.test_step_indices.append(indices)
        self.test_step_answers.extend(answers)
        return loss
