import tomllib

from src import ROOT_DIR

_cfg_path = ROOT_DIR / "configs" / "defaults.toml"
with open(_cfg_path, "rb") as _f:
    _cfg = tomllib.load(_f)
_configured_datasets = set(_cfg.get("datasets", {}).keys())
_discovered_datasets = {f.name.split(".")[0] for f in (ROOT_DIR / "data" / "datasets").glob("*.parquet")}
_dataset_choices = sorted(_configured_datasets | _discovered_datasets) or None


def arg_setup(parser):
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name defined in configs/defaults.toml under [profiles.<name>]. Values are applied unless overridden by CLI flags.",
    )
    parser.add_argument(
        "--processonly",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        default=False,
        help="With --processonly, only write split_dataset parquet (train/val/test/holdout); skip tokenized .pt files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=_dataset_choices,
        default="megavul",
        help="Dataset to use for training/testing.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Use CPU for training/testing.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of samples to process. If not specified, process all samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=None,
        help="Gradient accumulation steps (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with a smaller dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default from configs/defaults.toml). Mutually exclusive with --max-steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Total training steps. Mutually exclusive with --epochs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-1B",
        help="Model to use for training/testing.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reserved flag for explicit dataset reprocessing behavior.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optimizer weight decay (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--cw",
        type=int,
        default=None,
        help="Context window size (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--date",
        default="2023-04-01",
        type=str,
        help="Specific date to use for date-based split (format: YYYY-MM-DD).",
    )
    parser.add_argument(
        "--fold",
        default=None,
        type=int,
        help="Specific fold to use for cross-validation. If not specified, single fold is used.",
    )
    parser.add_argument(
        "--total_folds",
        default=10,
        type=int,
        help="Total number of folds for cross-validation.",
    )
    parser.add_argument(
        "--fold_type",
        choices=[
            "single_random",
            "single_date",
            "k_fold_cross_validation",
            "temporal_loo_block_cross_validation",
            "random_loo_block_cross_validation",
            "temporal_sliding_window_cross_validation",
            "random_sliding_window_cross_validation",
            "temporal_growing_window_cross_validation",
            "random_growing_window_cross_validation",
            "mixed_resampling_temporal_single",
            "mixed_resampling_random_single",
            "mixed_resampling_temporal_loo_block",
            "mixed_resampling_random_loo_block",
            "undersampling_random_single",
            "undersampling_temporal_single",
            "undersampling_random_loo_block",
            "undersampling_temporal_loo_block",
            "oversampling_random_single",
            "oversampling_temporal_single",
            "oversampling_random_loo_block",
            "oversampling_temporal_loo_block",
        ],
        default="single_random",
        help="Type of fold strategy to use.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Tag to identify the experiment (metadata only, does not select model family).",
    )
    parser.add_argument(
        "--no_holdout",
        action="store_true",
        help="Disable holdout set for testing.",
    )
    parser.add_argument(
        "--skip_fit",
        action="store_true",
        help="Skip the training phase and only run evaluation.",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run zero-shot evaluation only (no fine-tuning).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Base URL of OpenAI-compatible API. Optional for GGUF models: omit to auto-spawn a local llama-server.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for the local llama-server to become ready (default: 120).",
    )
    parser.add_argument(
        "--templated",
        action="store_true",
        help="Use the templated alpaca dataset instead of the raw dataset without any preprocessing.",
    )
    parser.add_argument(
        "--opro",
        action="store_true",
        help="Run OPRO prompt optimization before zero-shot evaluation. Requires --zero-shot.",
    )
    parser.add_argument(
        "--opro-steps",
        type=int,
        default=None,
        help="OPRO optimization steps (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--opro-train-size",
        type=int,
        default=None,
        help="OPRO training samples per step (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--opro-candidates",
        type=int,
        default=8,
        help="OPRO candidates generated per step (default from configs/defaults.toml).",
    )
    parser.add_argument(
        "--mixed-resampling-multiplier",
        type=float,
        default=None,
        help="Undersample multiplier for mixed_resampling fold types (default from defaults.toml).",
    )
    parser.add_argument(
        "--mixed-resampling-oversample-cap",
        type=float,
        default=None,
        help="Max fraction of existing train samples to add via oversampling (e.g. 2.0 = up to 200%% added). 0.0 disables oversampling (default from defaults.toml).",
    )
    parser.add_argument(
        "--undersampling-multiplier",
        type=float,
        default=None,
        help="Scale factor for CWE undersampling fold types (default from defaults.toml).",
    )
    parser.add_argument(
        "--oversampling-multiplier",
        type=float,
        default=None,
        help="Scale factor for CWE oversampling fold types (default from defaults.toml).",
    )
    parser.add_argument(
        "--cwe-assignment-strategy",
        type=str,
        default=None,
        choices=["val_most_common", "primary", "explode"],
        help="How to bucket each row for CWE resampling: 'val_most_common' picks the CWE most common in val from the row's pruned list; 'primary' picks the first CWE in the pruned list; 'explode' treats each row as a member of every CWE it lists (multi-bucket greedy allocation, per-row oversample cap) (default from defaults.toml).",
    )
    parser.add_argument(
        "--oversampling-cap",
        type=float,
        default=None,
        help="Max fraction of existing train samples to add via CWE oversampling (default from defaults.toml).",
    )
    parser.add_argument(
        "--holdout-prop",
        type=float,
        default=None,
        help="Fraction of total dataset held out for final evaluation (default from defaults.toml).",
    )
    parser.add_argument(
        "--test-prop",
        type=float,
        default=None,
        help="Fraction of non-holdout data used for val and test each (default from defaults.toml).",
    )

    return parser
