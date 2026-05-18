from typing import Dict

import torch

from src.datasets.datamodule import GenericDataset


class PlainTextDataset(GenericDataset):
    requires_tokenizer = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tokenize(self, source_code_text: str, label: int, index: int) -> Dict[str, torch.Tensor | int]:
        raise NotImplementedError("PlainTextDataset does not tokenize data")

    def _build_record(self, row: dict) -> dict:
        return {
            "text": row["func_before"],
            "label": row["vulnerability"],
            "index": row["index"],
        }
