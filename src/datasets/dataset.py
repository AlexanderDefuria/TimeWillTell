import abc
from enum import auto, Enum
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import data_dir, storage_structure
from src.models import get_hf_cache_dir
from src.datasets.mixed_resampling import mixed_resample_train_indices
from src.datasets.undersampling import undersample_train_indices, oversample_train_indices


class WindowType(Enum):
    GROWING = auto()
    SLIDING = auto()


class GenericDataset(Dataset, abc.ABC):
    requires_tokenizer: bool = True

    def __init__(
        self,
        dataset_file: Path,
        seed: int,
        instruction: str,
        tokenizer_name: str,
        max_length: int,
        tag: str,
        fold_type: str,
        size: Optional[int] = None,
        date: str = "random",
        fold: Optional[int] = None,
        total_folds: Optional[int] = None,
        test_prop: float = 0.1,
        holdout_prop: float = 0.1,
        stratify_by_cwe: bool = False,
        commit_based_random_splitting: bool = False,
        hold_out_test_set: bool = True,
        mixed_resampling_multiplier: float = 1.0,
        mixed_resampling_oversample_cap: float = 2.0,
        undersampling_multiplier: float = 1.0,
        oversampling_multiplier: float = 1.0,
        oversampling_cap: float = 2.0,
        cwe_assignment_strategy: str = "val_most_common",
    ):
        self.dataset_file = dataset_file
        self.hold_out_test_set_df: Optional[pl.DataFrame] = None
        self.instruction = instruction
        self.tokenizer_name = tokenizer_name
        if self.requires_tokenizer:
            self.tokenizer = self.get_tokenizer(tokenizer_name)
        else:
            self.tokenizer = None
        self.max_length = max_length
        self.size = size
        self.seed = seed
        self.date: str = date
        self.fold = fold
        self.total_folds = total_folds
        self.test_prop = test_prop
        self.holdout_prop = holdout_prop
        self.stratify_by_cwe = stratify_by_cwe
        self.commit_based_random_splitting = commit_based_random_splitting
        self.hold_out_test_set = hold_out_test_set
        self.mixed_resampling_multiplier = mixed_resampling_multiplier
        self.mixed_resampling_oversample_cap = mixed_resampling_oversample_cap
        self.undersampling_multiplier = undersampling_multiplier
        self.oversampling_multiplier = oversampling_multiplier
        self.oversampling_cap = oversampling_cap
        self.cwe_assignment_strategy = cwe_assignment_strategy
        self._pending_oversample_extras: list[int] = []
        self.tag = tag
        self.fold_type = fold_type
        self._fold_type_fns = {
            "single_random": self.get_random_train_val_test_split,
            "single_date": self.get_date_based_train_val_test_split,
            "k_fold_cross_validation": self.get_k_fold_train_val_test_split,
            "random_loo_block_cross_validation": self.get_random_loo_block_train_val_test_split,
            "temporal_loo_block_cross_validation": self.get_temporal_loo_block_train_val_test_split,
            "random_sliding_window_cross_validation": self.get_random_sliding_window_train_val_test_split,
            "temporal_sliding_window_cross_validation": self.get_temporal_sliding_window_train_val_test_split,
            "random_growing_window_cross_validation": self.get_random_growing_window_train_val_test_split,
            "temporal_growing_window_cross_validation": self.get_temporal_growing_window_train_val_test_split,
            "balanced_temporal_loo_block_cross_validation": self.get_temporal_balanced_loo_block_train_val_test_split,
            "balanced_random_loo_block_cross_validation": self.get_random_balanced_loo_block_train_val_test_split,
            "mixed_resampling_temporal_single": self.get_mixed_resampling_temporal_single_split,
            "mixed_resampling_random_single": self.get_mixed_resampling_random_single_split,
            "mixed_resampling_temporal_loo_block": self.get_mixed_resampling_temporal_loo_block_split,
            "mixed_resampling_random_loo_block": self.get_mixed_resampling_random_loo_block_split,
            "undersampling_random_single": self.get_undersampling_random_single_split,
            "undersampling_temporal_single": self.get_undersampling_temporal_single_split,
            "undersampling_random_loo_block": self.get_undersampling_random_loo_block_split,
            "undersampling_temporal_loo_block": self.get_undersampling_temporal_loo_block_split,
            "oversampling_random_single": self.get_oversampling_random_single_split,
            "oversampling_temporal_single": self.get_oversampling_temporal_single_split,
            "oversampling_random_loo_block": self.get_oversampling_random_loo_block_split,
            "oversampling_temporal_loo_block": self.get_oversampling_temporal_loo_block_split,
        }
        assert self.fold_type in self._fold_type_fns, f"Fold type {self.fold_type} is not valid. Must be one of {list(self._fold_type_fns.keys())}."
        if self.date != "random":
            datetime.strptime(self.date, "%Y-%m-%d")
        os.makedirs(self.processed_dir, exist_ok=True)

    @property
    def processed_dir(self) -> Path:
        dir = storage_structure(
            root=data_dir() / "processed",
            tag=self.tag,
            dataset=f"{self.dataset_file.stem}",
            dataset_class=type(self).__name__,
            fold_type=self.fold_type,
            date=self.date,
            fold=self.fold,
            total_fold=self.total_folds,
            model_name=self.tokenizer_name.split("/")[-1],
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )

        return dir

    def __len__(self) -> int:
        return len(self.tokenized_data["train"]) + len(self.tokenized_data["val"]) + len(self.tokenized_data["test"])

    @abc.abstractmethod
    def tokenize(
        self,
        source_code_text: str,
        label: int,
        index: int,
    ) -> Dict[str, torch.Tensor | int]:
        raise NotImplementedError

    def tokenized_file(self, phase) -> Path:
        return self.processed_dir / f"tokenized_{self.max_length}_{self.size}_s{self.seed}_{phase}.pt"

    @property
    def processed_file(self) -> Path:
        return self.processed_dir / f"split_dataset_{self.size}_s{self.seed}.parquet"

    def split_dataset(self, hold_out_test_set: bool = True, df_override: Optional[pl.DataFrame] = None):
        if df_override is None:
            print(f"Processing dataset from file: {self.dataset_file}")
            df = pl.read_parquet(self.dataset_file)
        else:
            print(f"Processing dataset from provided DataFrame override")
            df = df_override
        entire_df_size = len(df)

        # If using a hold-out test set, remove the most recent holdout_prop of data based on publish_date
        # This will be used to evaluate all the folds later on to have a consistent test set
        if hold_out_test_set:
            hold_out_size = int(entire_df_size * self.holdout_prop)
            hold_out_indices = df.top_k(hold_out_size, by="publish_date")["index"].to_list()
            hold_out_df = df.filter(pl.col("index").is_in(hold_out_indices))
            hold_out_df = hold_out_df.with_columns(pl.lit("holdout").alias("phase"))
            df = df.filter(~pl.col("index").is_in(hold_out_indices))
            print(f"Max Train/Test Date {df['publish_date'].max()}")
            print(f"Min Hold-out Date {hold_out_df['publish_date'].min()}")
            print(f"Holding out {len(hold_out_indices)} samples for hold-out test set.")
        else:
            hold_out_indices = []
            hold_out_df = pl.DataFrame()

        # Get train, val, test splits using the function for this fold type
        train_indices, val_indices, test_indices = self._fold_type_fns[self.fold_type](df)
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}, Hold-out size: {len(hold_out_indices)}")

        # Ensure that all indices are accounted for
        df = df.with_columns(
            pl.when(pl.col("index").is_in(train_indices))
            .then(pl.lit("train"))
            .otherwise(
                pl.when(pl.col("index").is_in(val_indices))
                .then(pl.lit("val"))
                .otherwise(
                    pl.when(pl.col("index").is_in(test_indices))
                    .then(pl.lit("test"))
                    .otherwise(
                        pl.lit("unknown"),
                    ),
                )
            )
            .alias("phase"),
        )
        size = len(df) if self.size is None or self.size == -1 else self.size
        df = df.sample(n=size, seed=self.seed, shuffle=True)  # Shuffle the dataset using the constant seed

        # Inject oversampled duplicate rows for mixed_resampling splits with oversample_cap > 0
        extras = self._pending_oversample_extras
        self._pending_oversample_extras = []
        if extras:
            source_df = df_override if df_override is not None else pl.read_parquet(self.dataset_file)
            extra_rows = source_df.filter(pl.col("index").is_in(extras))
            # Synthetic indices must clear the full source (including holdout, which was
            # already filtered out of `df`), otherwise they collide with holdout indices.
            max_idx = max(source_df["index"].max(), df["index"].max())
            synthetic_indices = list(range(max_idx + 1, max_idx + 1 + len(extras)))
            # Build the extras dataframe preserving original row order from the extras list
            index_to_row = {r["index"]: r for r in extra_rows.iter_rows(named=True)}
            rows = [index_to_row[i] for i in extras if i in index_to_row]
            extras_df = pl.DataFrame(rows, schema=source_df.schema)
            extras_df = extras_df.with_columns([
                pl.Series("index", synthetic_indices, dtype=df["index"].dtype),
                pl.lit("train").alias("phase"),
            ])
            df = pl.concat([df, extras_df.select(df.columns)])
            print(f"Appended {len(extras_df)} oversampled duplicate train rows.")

        # Add back hold-out set if applicable
        if len(hold_out_indices) > 0:
            df = pl.concat([df, hold_out_df])
        df = df.select(["vulnerability", "func_before", "cwe_ids", "phase", "index"])

        n_extras = len(extras) if extras else 0
        assert len(df) == size + n_extras + len(hold_out_df), f"Dataset size {len(df)} does not match expected size {size + n_extras + len(hold_out_df)}"
        assert len(set(train_indices).intersection(set(val_indices))) == 0
        assert len(set(train_indices).intersection(set(test_indices))) == 0
        assert len(set(val_indices).intersection(set(test_indices))) == 0
        assert len(set(hold_out_indices).intersection(set(train_indices))) == 0
        assert len(set(hold_out_indices).intersection(set(val_indices))) == 0
        assert len(set(hold_out_indices).intersection(set(test_indices))) == 0
        os.makedirs(self.processed_file.parent, exist_ok=True)
        df.write_parquet(self.processed_file, compression="gzip")

    def get_split_dataset(self, hold_out_test_set: bool = True, df_override: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        print(f"Loading processed data from {self.processed_file}")
        if not self.processed_file.exists():
            self.split_dataset(df_override=df_override, hold_out_test_set=hold_out_test_set)
        df = pl.read_parquet(self.processed_file)
        if self.size is not None and self.size != df.height:
            print(f"Resplitting dataset to size {self.size}")
            self.split_dataset(df_override=df_override, hold_out_test_set=hold_out_test_set)
            df = pl.read_parquet(self.processed_file)
        return df

    def _build_record(self, row: dict) -> dict:
        """Build a single data record from a DataFrame row. Override for custom formats."""
        return self.tokenize(
            source_code_text=row["func_before"],
            label=row["vulnerability"],
            index=row["index"],
        )

    def process(self):
        """
        Process the dataset and save the tokenized data.
        """
        df = self.get_split_dataset(hold_out_test_set=self.hold_out_test_set)
        self.tokenized_data = {}
        for phase in ["train", "val", "test", "holdout"]:
            if self.tokenized_file(phase).exists():
                self.tokenized_data[phase] = torch.load(open(self.tokenized_file(phase), "rb"))
            else:
                data = []
                phase_df = df.filter(pl.col("phase") == phase)
                for row in tqdm(
                    phase_df.iter_rows(named=True),
                    desc=f"Processing {phase}",
                    total=phase_df.height,
                ):
                    data.append(self._build_record(row))
                os.makedirs(self.processed_file.parent, exist_ok=True)
                torch.save(data, open(self.tokenized_file(phase), "wb"))
                self.tokenized_data[phase] = data
            print(f"Loaded phase: {phase} - self.tokenized_file(phase)")

    @staticmethod
    def get_tokenizer(tokenizer_name: str) -> Callable:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=get_hf_cache_dir(), offline=True)
        if (
            "llama" in tokenizer_name.lower()
            or "mistral" in tokenizer_name.lower()
            or "phi" in tokenizer_name.lower()
            or "star" in tokenizer_name.lower()
        ):
            print("Using Llama tokenizer with special tokens.")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        return tokenizer

    def _validate_fold_params(self, min_folds: int = 2) -> None:
        """Validate fold and total_folds parameters."""
        assert self.fold is not None and self.total_folds is not None, "Fold and total_folds must be specified."
        assert 0 <= self.fold < self.total_folds, f"Fold {self.fold} must be between 0 and {self.total_folds - 1}."
        assert self.total_folds >= min_folds, f"Total folds must be at least {min_folds}."
        assert isinstance(self.total_folds, int), "Total folds must be an integer."
        assert isinstance(self.fold, int), "Fold must be an integer."

    def get_random_indices(self, df: pl.DataFrame) -> np.ndarray:
        np.random.seed(self.seed)
        if self.commit_based_random_splitting:
            commit_ids = df["commit_hash"].unique().to_list()
            np.random.shuffle(commit_ids)
            commit_id_to_indices = {cid: df.filter(pl.col("commit_hash") == cid)["index"].to_list() for cid in commit_ids}
            indices = []
            for cid in commit_ids:
                indices.extend(commit_id_to_indices[cid])
            indices = np.array(indices)
        else:
            indices = df.sort(by="date")["index"].to_numpy()
            indices = np.random.permutation(indices)
        return indices

    def get_random_train_val_test_split(self, df: pl.DataFrame):
        indices = self.get_random_indices(df)
        test_size = int(len(df) * self.test_prop)
        val_size = test_size
        train_size = len(df) - test_size - val_size
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size : train_size + val_size].tolist()
        test_indices = indices[train_size + val_size :].tolist()
        return train_indices, val_indices, test_indices

    def get_date_based_train_val_test_split(self, df: pl.DataFrame):
        split_date = datetime.strptime(self.date, "%Y-%m-%d").date()
        train_indices = df.filter(pl.col("date") < split_date).get_column("index").to_list()
        test_val_indices = df.filter(pl.col("date") >= split_date).get_column("index").to_list()
        np.random.seed(self.seed)
        np.random.shuffle(test_val_indices)
        test_size = int(len(test_val_indices) * 0.5)
        val_size = len(test_val_indices) - test_size
        val_indices = test_val_indices[:val_size]
        test_indices = test_val_indices[val_size:]
        return train_indices, val_indices, test_indices

    def get_k_fold_train_val_test_split(self, df: pl.DataFrame):
        self._validate_fold_params(min_folds=2)

        np.random.seed(self.seed)
        # Must be writable: Polars may expose the underlying buffer as read-only.
        indices = df["index"].to_numpy().copy()
        np.random.shuffle(indices)
        fold_sizes = np.full(self.total_folds, len(df) // self.total_folds, dtype=int)
        fold_sizes[: len(df) % self.total_folds] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop].tolist())
            current = stop
        test_indices = folds[self.fold]
        val_fold = (self.fold + 1) % self.total_folds
        val_indices = folds[val_fold]
        train_indices = [idx for i, fold in enumerate(folds) if i != self.fold and i != val_fold for idx in fold]
        print(f"Test Size: {len(test_indices)}, Val Size: {len(val_indices)}, Train Size: {len(train_indices)}")
        return train_indices, val_indices, test_indices

    def get_random_loo_block_train_val_test_split(self, df: pl.DataFrame):
        indices = self.get_random_indices(df)
        return self.get_loo_block_train_val_test_split(df, indices)

    def get_temporal_loo_block_train_val_test_split(self, df: pl.DataFrame):
        np.random.seed(self.seed)
        indices = df.sort(by="date")["index"].to_numpy()
        return self.get_loo_block_train_val_test_split(df, indices)

    def get_loo_block_train_val_test_split(self, df: pl.DataFrame, indices):
        self._validate_fold_params(min_folds=3)

        block_size = len(df) // self.total_folds
        test_start = self.fold * block_size
        test_end = (self.fold + 1) * block_size if self.fold != self.total_folds - 1 else len(df)
        test_indices = indices[test_start:test_end].tolist()
        train_val_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        val_block_size = len(train_val_indices) // (self.total_folds - 1)

        # Sample validation randomly from remaining indices (not block level)
        val_indices = np.random.choice(
            train_val_indices,
            size=val_block_size,
            replace=False,
        ).tolist()
        val_indices_set = set(val_indices)
        train_indices_set = set(train_val_indices) - val_indices_set
        train_indices = list(train_indices_set)

        return train_indices, val_indices, test_indices

    def get_random_sliding_window_train_val_test_split(self, df: pl.DataFrame):
        """Sliding window begins with minimum training size of 50% of data, then moves forward by test size each fold."""
        indices = self.get_random_indices(df)
        window_type = WindowType.SLIDING
        return self.get_window_train_val_test_split(df, indices, window_type)

    def get_temporal_sliding_window_train_val_test_split(self, df: pl.DataFrame):
        """Sliding window begins with minimum training size of 50% of data, then moves forward by test size each fold."""
        indices = df.sort(by="date")["index"].to_numpy()
        window_type = WindowType.SLIDING
        return self.get_window_train_val_test_split(df, indices, window_type)

    def get_random_growing_window_train_val_test_split(self, df: pl.DataFrame):
        indices = self.get_random_indices(df)
        window_type = WindowType.GROWING
        return self.get_window_train_val_test_split(df, indices, window_type)

    def get_temporal_growing_window_train_val_test_split(self, df: pl.DataFrame):
        indices = df.sort(by="date")["index"].to_numpy()
        window_type = WindowType.GROWING
        return self.get_window_train_val_test_split(df, indices, window_type)

    def get_temporal_balanced_loo_block_train_val_test_split(self, df: pl.DataFrame):
    # def get_temporal_loo_block_train_val_test_split(self, df: pl.DataFrame):
        np.random.seed(self.seed)
        indices = df.sort(by="date")["index"].to_numpy()
    
        train_indices, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)

        val_df = df.filter(pl.col("index").is_in(val_indices))
        test_df = df.filter(pl.col("index").is_in(test_indices))
        train_df = df.filter(pl.col("index").is_in(train_indices))

        # Check class balance in each set
        # 1. rollup remove parent CWE's
        # 2. get class distribution in val and test sets
        # 3. undersample from train set to match val distribution

        raise NotImplementedError("Balanced LOO block CV is not yet implemented. This function currently returns the same splits as temporal LOO block CV without true balancing.")


    def get_random_balanced_loo_block_train_val_test_split(self, df: pl.DataFrame):
        # Get the correct size from the temporal balanced split to ensure the same number of samples in train
        # Then randomly undersample the random loo block split to that size to get a balanced random split
        # This works because the random loo block split will be inherently balanced.
        indices = self.get_temporal_balanced_loo_block_train_val_test_split(df)
        train_size = len(indices[0])
        train_indices, val_indices, test_indices = self.random_loo_block_cross_validation(df)
        train_indices = np.random.choice(train_indices, size=train_size, replace=False).tolist()
        return train_indices, val_indices, test_indices

    def get_window_train_val_test_split(self, df: pl.DataFrame, indices, window_type: WindowType):
        self._validate_fold_params(min_folds=3)

        np.random.seed(self.seed)
        minimum_train_size = 0.5
        val_size = 0.1
        test_size = 0.4 / self.total_folds
        total_size = len(df)

        train_end = int(total_size * (minimum_train_size + self.fold * test_size))
        val_start = train_end
        val_end = val_start + int(total_size * val_size)
        test_start = val_end
        test_end = test_start + int(total_size * test_size)
        if window_type == WindowType.SLIDING:
            train_start = int(train_end - minimum_train_size * total_size)
        else:
            train_start = 0

        train_indices = indices[train_start:train_end].tolist()
        val_indices = indices[val_start:val_end].tolist()
        test_indices = indices[test_start:test_end].tolist()

        return train_indices, val_indices, test_indices

    def get_mixed_resampling_temporal_single_split(self, df: pl.DataFrame):
        """Temporal 90/10 train-pool/test split with val randomly sampled from the train pool.

        Test = most recent test_prop of data by date.
        Val = random sample of size test_n drawn from the remaining 90% (train pool).
        Train (raw) = train pool minus val, then CWE mixed-resampled against val distribution.
        """
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = mixed_resample_train_indices(
            df, train_indices_raw, val_indices,
            self.mixed_resampling_multiplier, self.seed,
            oversample_cap=self.mixed_resampling_oversample_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_mixed_resampling_random_single_split(self, df: pl.DataFrame):
        """Mixed-resampling split with randomly sampled val, then shuffle all indices.

        Sizes are determined by the same 90/10 temporal procedure as the temporal
        variant; the resulting indices are then shuffled to destroy temporal ordering.
        """
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = mixed_resample_train_indices(
            df, train_indices_raw, val_indices,
            self.mixed_resampling_multiplier, self.seed,
            oversample_cap=self.mixed_resampling_oversample_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )

        combined_indices = train_indices + val_indices + test_indices
        np.random.seed(self.seed)
        combined_indices = np.random.permutation(combined_indices).tolist()

        n_train = len(train_indices)
        n_val = len(val_indices)
        n_test = len(test_indices)

        train_final = combined_indices[:n_train]
        val_final = combined_indices[n_train : n_train + n_val]
        test_final = combined_indices[n_train + n_val : n_train + n_val + n_test]

        return train_final, val_final, test_final

    def get_mixed_resampling_temporal_loo_block_split(self, df: pl.DataFrame):
        """Temporal LOO block CV with train resampled to match the test block's CWE distribution."""
        np.random.seed(self.seed)
        indices = df.sort(by="date")["index"].to_numpy()
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = mixed_resample_train_indices(
            df, train_indices_raw, test_indices,
            self.mixed_resampling_multiplier, self.seed,
            oversample_cap=self.mixed_resampling_oversample_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_mixed_resampling_random_loo_block_split(self, df: pl.DataFrame):
        """Random LOO block CV with train resampled to match the test block's CWE distribution."""
        np.random.seed(self.seed)
        indices = self.get_random_indices(df)
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = mixed_resample_train_indices(
            df, train_indices_raw, test_indices,
            self.mixed_resampling_multiplier, self.seed,
            oversample_cap=self.mixed_resampling_oversample_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_undersampling_temporal_single_split(self, df: pl.DataFrame):
        """Temporal 90/10 split with train CWE-undersampled to match val distribution."""
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = undersample_train_indices(
            df, train_indices_raw, val_indices, self.undersampling_multiplier, self.seed,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_undersampling_random_single_split(self, df: pl.DataFrame):
        """Random 90/10 split with train CWE-undersampled to match val distribution."""
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = undersample_train_indices(
            df, train_indices_raw, val_indices, self.undersampling_multiplier, self.seed,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        combined_indices = train_indices + val_indices + test_indices
        np.random.seed(self.seed)
        combined_indices = np.random.permutation(combined_indices).tolist()
        n_train = len(train_indices)
        n_val = len(val_indices)
        return combined_indices[:n_train], combined_indices[n_train:n_train + n_val], combined_indices[n_train + n_val:]

    def get_undersampling_temporal_loo_block_split(self, df: pl.DataFrame):
        """Temporal LOO block CV with train CWE-undersampled to match val distribution."""
        np.random.seed(self.seed)
        indices = df.sort(by="date")["index"].to_numpy()
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = undersample_train_indices(
            df, train_indices_raw, val_indices, self.undersampling_multiplier, self.seed,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_undersampling_random_loo_block_split(self, df: pl.DataFrame):
        """Random LOO block CV with train CWE-undersampled to match val distribution."""
        np.random.seed(self.seed)
        indices = self.get_random_indices(df)
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = undersample_train_indices(
            df, train_indices_raw, val_indices, self.undersampling_multiplier, self.seed,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_oversampling_temporal_single_split(self, df: pl.DataFrame):
        """Temporal 90/10 split with train CWE-oversampled to match val distribution."""
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = oversample_train_indices(
            df, train_indices_raw, val_indices,
            self.oversampling_multiplier, self.seed, self.oversampling_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_oversampling_random_single_split(self, df: pl.DataFrame):
        """Random 90/10 split with train CWE-oversampled to match val distribution."""
        sorted_df = df.sort("date")
        n = len(sorted_df)
        test_n = int(n * self.test_prop)
        test_start = n - test_n
        test_indices = sorted_df[test_start:]["index"].to_list()
        train_pool = sorted_df[:test_start]["index"].to_list()
        rng = np.random.default_rng(self.seed)
        val_indices = rng.choice(train_pool, size=test_n, replace=False).tolist()
        val_set = set(val_indices)
        train_indices_raw = [i for i in train_pool if i not in val_set]
        train_indices, self._pending_oversample_extras = oversample_train_indices(
            df, train_indices_raw, val_indices,
            self.oversampling_multiplier, self.seed, self.oversampling_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        combined_indices = train_indices + val_indices + test_indices
        np.random.seed(self.seed)
        combined_indices = np.random.permutation(combined_indices).tolist()
        n_train = len(train_indices)
        n_val = len(val_indices)
        return combined_indices[:n_train], combined_indices[n_train:n_train + n_val], combined_indices[n_train + n_val:]

    def get_oversampling_temporal_loo_block_split(self, df: pl.DataFrame):
        """Temporal LOO block CV with train CWE-oversampled to match val distribution."""
        np.random.seed(self.seed)
        indices = df.sort(by="date")["index"].to_numpy()
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = oversample_train_indices(
            df, train_indices_raw, val_indices,
            self.oversampling_multiplier, self.seed, self.oversampling_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices

    def get_oversampling_random_loo_block_split(self, df: pl.DataFrame):
        """Random LOO block CV with train CWE-oversampled to match val distribution."""
        np.random.seed(self.seed)
        indices = self.get_random_indices(df)
        train_indices_raw, val_indices, test_indices = self.get_loo_block_train_val_test_split(df, indices)
        train_indices, self._pending_oversample_extras = oversample_train_indices(
            df, train_indices_raw, val_indices,
            self.oversampling_multiplier, self.seed, self.oversampling_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        return train_indices, val_indices, test_indices


