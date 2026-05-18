import tomllib
import urllib.request
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
import polars as pl
from tqdm import tqdm

from src.datasets.dataset import GenericDataset
from src.utils import data_dir
from src import ROOT_DIR


def _load_dataset_config(name: str) -> dict | None:
    cfg_path = ROOT_DIR / "configs" / "defaults.toml"
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("datasets", {}).get(name)


def _download_dataset(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class GenericDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: str,
        max_length: int,
        instruction: str,
        batch_size: int,
        fold_type: str,
        dataset_class: type[GenericDataset],
        tag: str,
        seed: int = 42,
        dataset_size=None,
        workers: int = 1,
        date: str = "random",
        debug: bool = False,
        fold: Optional[int] = None,
        total_folds: Optional[int] = None,
        test_prop: float = 0.1,
        holdout_prop: float = 0.1,
        commit_based_random_splitting: bool = False,
        hold_out_test_set: bool = True,
        mixed_resampling_multiplier: float = 1.0,
        mixed_resampling_oversample_cap: float = 2.0,
        undersampling_multiplier: float = 1.0,
        oversampling_multiplier: float = 1.0,
        oversampling_cap: float = 2.0,
        cwe_assignment_strategy: str = "val_most_common",
    ):
        """
        Function Source Code Data Module for loading the dataset.
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length
        self.workers = workers
        self.instruction = instruction
        self.date = date
        self.dataset_class = dataset_class
        self.debug = debug
        self.fold = fold
        self.total_folds = total_folds
        self.test_prop = test_prop
        self.holdout_prop = holdout_prop
        self.fold_type = fold_type
        self.tag = tag
        self.commit_based_random_splitting = commit_based_random_splitting
        self.hold_out_test_set = hold_out_test_set
        self.mixed_resampling_multiplier = mixed_resampling_multiplier
        self.mixed_resampling_oversample_cap = mixed_resampling_oversample_cap
        self.undersampling_multiplier = undersampling_multiplier
        self.oversampling_multiplier = oversampling_multiplier
        self.oversampling_cap = oversampling_cap
        self.cwe_assignment_strategy = cwe_assignment_strategy

    def dataset_source(self) -> Path:
        return Path(data_dir()) / "datasets" / f"{self.dataset_name}.parquet"

    def prepare_data(self) -> None:
        """
        Make sure that the dataset is in parquet format.
        """
        if self.dataset_source().exists():
            print(f"Dataset source {self.dataset_source()} exists.")
            return

        json_path = self.dataset_source().with_suffix(".json")
        csv_path = self.dataset_source().with_suffix(".csv")
        if json_path.exists() and csv_path.exists():
            raise ValueError(f"Both {json_path} and {csv_path} exist. Please remove one of them.")
        elif not json_path.exists() and not csv_path.exists():
            dataset_cfg = _load_dataset_config(self.dataset_name)
            url = (dataset_cfg or {}).get("zenodo_url")
            if url and "PLACEHOLDER" not in url:
                print(f"Downloading {self.dataset_name} from {url} ...")
                _download_dataset(url, self.dataset_source())
                return
            raise FileNotFoundError(f"Neither {json_path} nor {csv_path} exist. Please provide one of them.")

        # Selecting and renaming columns based on dataset
        print(f"Formatting dataset {self.dataset_name}.")
        if "megavul" in self.dataset_name:
            import pandas as pd
            df = pd.read_json(json_path)[
                [
                    "publish_date",
                    "commit_hash",
                    "repo_name",
                    "is_vul",
                    "git_url",
                    "func_before",
                    "func",
                    "cwe_ids",
                    "file_path",
                    "func_name",
                ]
            ]
            df = pl.from_pandas(df)
            assert type(df) == pl.DataFrame
            columns = [
                pl.col("publish_date").str.to_datetime().alias("date"),
                pl.col("commit_hash").str.replace(r"^0x", "", literal=True).alias("commit"),
                pl.col("repo_name").str.replace(r"^0x", "", literal=True).alias("project"),
                pl.col("is_vul").cast(int).alias("vulnerability"),
                pl.col("file_path").str.replace(r"^0x", "", literal=True).alias("file_path"),
                pl.col("git_url").str.replace(r"^0x", "", literal=True).alias("code_link"),
                # pl.col("func").str.replace(r"^0x", "", literal=True).alias("func_before"),
                pl.when(pl.col("func_before").is_null())
                .then(pl.col("func"))
                .otherwise(
                    pl.col("func_before"),
                )
                .str.replace(r"^0x", "", literal=True)
                .alias("func_before_real"),
                pl.col("cwe_ids").alias("cwe_ids"),
                pl.col("func_name").str.replace(r"^0x", "", literal=True).alias("func_name"),
            ]
            df = df.with_columns(columns).with_row_index()
            df = df.drop(["func_before", "func"]).rename({"func_before_real": "func_before"})
            df.write_parquet(self.dataset_source())
        elif "devign" in self.dataset_name or "bigvul" in self.dataset_name:
            df = pl.read_csv(csv_path)
            columns = [
                pl.col("sha_id").str.replace(r"^0x", "", literal=True).alias("commit"),
                pl.col("project").str.replace(r"^0x", "", literal=True).alias("project"),
                pl.col("codeLink").str.replace(r"^0x", "", literal=True).alias("code_link"),
                pl.col("vulnerability").alias("vulnerability"),
                pl.col("func_before").str.replace(r"^0x", "", literal=True).alias("func_before"),
                pl.lit("N/A").alias("cwe_ids"),
            ]
            df = df.with_columns(columns).with_row_index()
            df.write_parquet(self.dataset_source())
        else:
            raise ValueError(f"Dataset {self.dataset_name} not recognized. Please provide a valid dataset.")

    def setup(self, stage=None, process=False, splits_only: bool = False):
        dataset = self.dataset_class(
            tokenizer_name=self.tokenizer,
            dataset_file=self.dataset_source(),
            instruction=self.instruction,
            seed=self.seed,
            tag=self.tag,
            size=self.dataset_size,
            max_length=self.max_length,
            date=self.date,
            fold=self.fold,
            total_folds=self.total_folds,
            test_prop=self.test_prop,
            holdout_prop=self.holdout_prop,
            fold_type=self.fold_type,
            commit_based_random_splitting=self.commit_based_random_splitting,
            hold_out_test_set=self.hold_out_test_set,
            mixed_resampling_multiplier=self.mixed_resampling_multiplier,
            mixed_resampling_oversample_cap=self.mixed_resampling_oversample_cap,
            undersampling_multiplier=self.undersampling_multiplier,
            oversampling_multiplier=self.oversampling_multiplier,
            oversampling_cap=self.oversampling_cap,
            cwe_assignment_strategy=self.cwe_assignment_strategy,
        )
        if splits_only:
            dataset.get_split_dataset(hold_out_test_set=self.hold_out_test_set)
            self.dataset = dataset
            self.train_data = None
            self.val_data = None
            self.test_data = None
            self.holdout_data = None
        else:
            dataset.process()
            self.dataset = dataset
            self.train_data = dataset.tokenized_data["train"]
            self.val_data = dataset.tokenized_data["val"]
            self.test_data = dataset.tokenized_data["test"]
            self.holdout_data = dataset.tokenized_data.get("holdout", None)

    def _make_loader(self, data, stateful: bool = False, shuffle: bool = False, collate_fn=None):
        loader_class = StatefulDataLoader if stateful else DataLoader
        return loader_class(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def train_dataloader(self, collate_fn=None):
        assert self.train_data is not None, "train_data is not set. Call setup() first."
        return self._make_loader(self.train_data, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self, collate_fn=None):
        assert self.val_data is not None, "val_data is not set. Call setup() first."
        return self._make_loader(self.val_data, stateful=True, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=None):
        assert self.test_data is not None, "test_data is not set. Call setup() first."
        return self._make_loader(self.test_data, stateful=True, collate_fn=collate_fn)

    def holdout_dataloader(self, collate_fn=None):
        assert self.holdout_data is not None, "holdout_data is not set. Call setup() first."
        return self._make_loader(self.holdout_data, stateful=True, collate_fn=collate_fn)
