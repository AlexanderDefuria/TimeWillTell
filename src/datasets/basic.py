import torch
from typing import Dict

from src.datasets.datamodule import GenericDataset


class RawDataset(GenericDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def tokenize(
        self,
        source_code_text: str,
        label: int,
        index: int,
    ) -> Dict[str, torch.Tensor | int]:

        tokenized = self.tokenizer(
            source_code_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label": label,
            "index": index,
        }
