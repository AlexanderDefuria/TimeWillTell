from typing import Dict
import torch

from src.datasets.datamodule import GenericDataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class AlpacaDataset(GenericDataset):
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

        alpaca_inputs = {
            "instruction": self.instruction,
            "input": source_code_text,
        }
        prompt = PROMPT_DICT["prompt_input"].format_map(alpaca_inputs)
        tokenized = self.tokenizer(
            prompt,
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
