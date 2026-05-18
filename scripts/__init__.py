from pathlib import Path


_SCRIPTS_DIR = Path(__file__).parent

PROJECT_ROOT = _SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EDA_DIR = DATA_DIR / "eda"

TB_LOGS_DIR = DATA_DIR / "tb_logs"
ALL_RESULTS_PARQUET = EDA_DIR / "all_tensorboard_results.csv"
TEST_RESULTS_CSV = EDA_DIR / "tensorboard_results.csv"
INDEX_RESULTS_FILE = EDA_DIR / "index_results.parquet"

MEGAVUL_DATASET_JSON = DATASETS_DIR / "megavul.json"
MEGAVUL_DATASET_PARQUET = DATASETS_DIR / "megavul.parquet"
BIGVUL_DATASET_CSV = DATASETS_DIR / "bigvul.csv"
DEVIGN_DATASET_PARQUET = DATASETS_DIR / "devign.parquet"

PRIMARY_FOLD_TYPES = [
    "k_fold_cross_validation",
    "random_growing_window_cross_validation",
    "random_loo_block_cross_validation",
    "temporal_growing_window_cross_validation",
    "temporal_loo_block_cross_validation",
]

CWE_RESAMPLING_FOLD_TYPES = [
    "undersampling_random_single",
    "undersampling_temporal_single",
    "undersampling_random_loo_block",
    "undersampling_temporal_loo_block",
    "oversampling_random_single",
    "oversampling_temporal_single",
    "oversampling_random_loo_block",
    "oversampling_temporal_loo_block",
    "mixed_resampling_random_single",
    "mixed_resampling_temporal_single",
    "mixed_resampling_random_loo_block",
    "mixed_resampling_temporal_loo_block",
]

CWE_RESAMPLING_BASELINES = [
    "random_loo_block_cross_validation",
    "temporal_loo_block_cross_validation",
]

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATASETS_DIR",
    "PROCESSED_DATA_DIR",
    "EDA_DIR",
    "TB_LOGS_DIR",
    "ALL_RESULTS_PARQUET",
    "TEST_RESULTS_CSV",
    "INDEX_RESULTS_FILE",
    "MEGAVUL_DATASET_JSON",
    "MEGAVUL_DATASET_PARQUET",
    "BIGVUL_DATASET_CSV",
    "DEVIGN_DATASET_PARQUET",
    "PRIMARY_FOLD_TYPES",
    "CWE_RESAMPLING_FOLD_TYPES",
    "CWE_RESAMPLING_BASELINES",
]
