from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    processonly: bool = False
    splits_only: bool = False
    cpu: bool = False
    dev: bool = False
    seed: int | None = None
    tag: str = "default"
    skip_fit: bool = False
    zero_shot: bool = False
    api_url: str | None = None
    api_timeout: int = 120
    reprocess: bool = False
    profile: str | None = None
    opro: bool = False
    opro_steps: int = 10
    opro_train_size: int = 128
    opro_candidates: int = 8


@dataclass
class DataConfig:
    dataset: str = "megavul"
    n: int | None = None
    batch_size: int | None = None
    cw: int = 16384
    date: str = "2023-04-01"
    fold: int | None = None
    total_folds: int = 10
    fold_type: str = "single_random"
    no_holdout: bool = False
    templated: bool = False
    mixed_resampling_multiplier: float = 1.0
    mixed_resampling_oversample_cap: float = 2.0
    undersampling_multiplier: float = 1.0
    oversampling_multiplier: float = 1.0
    oversampling_cap: float = 2.0
    cwe_assignment_strategy: str = "val_most_common"
    holdout_prop: float = 0.1
    test_prop: float = 0.1


@dataclass
class TrainerConfig:
    grad_acc: int = 1
    epochs: int = 3
    max_steps: int | None = None
    weight_decay: float = 0.01
    validation_interval: int = 1 # Validate every N epochs (default: 1, i.e. every epoch)


@dataclass
class ExperimentConfig:
    model: str = "llama3.2-1B"
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def resolved_skip_fit(self) -> bool:
        return self.run.skip_fit or self.run.zero_shot

    def resolved_no_holdout(self) -> bool:
        return self.data.no_holdout

    def validate(self) -> None:
        if self.run.splits_only and not self.run.processonly:
            raise ValueError("--splits-only requires --processonly.")
        if self.data.total_folds <= 1:
            raise ValueError("Total folds must be greater than 1.")
        if self.data.cw is not None and self.data.cw <= 0:
            raise ValueError("Context window (--cw) must be a positive integer.")
        if "_" in self.run.tag:
            raise ValueError("Tag cannot contain underscores '_', use '-' instead.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_resolved_dict(self) -> dict[str, Any]:
        base = self.as_dict()
        base["derived"] = {
            "skip_fit": self.resolved_skip_fit(),
            "no_holdout": self.resolved_no_holdout(),
        }
        return base


def _collect_cli_override_keys(parser) -> set[str]:
    overrides: set[str] = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=")[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            overrides.add(action.dest)
    return overrides


def load_profile_values(profile_path: str | None) -> dict[str, Any]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file '{profile_path}' does not exist.")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    # Supported format:
    # [cli]
    # model = "llama3.2-1B"
    # batch_size = 8
    if "cli" in raw and isinstance(raw["cli"], dict):
        return raw["cli"]
    return raw


def load_named_profile(name: str, defaults_path: Path) -> dict[str, Any]:
    if not defaults_path.exists():
        raise FileNotFoundError(f"defaults.toml not found at '{defaults_path}'.")
    with defaults_path.open("rb") as f:
        raw = tomllib.load(f)
    profiles = raw.get("profiles", {})
    if name not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Profile '{name}' not found in defaults.toml. Available: {available}")
    return profiles[name]


def _merge_profile_into_namespace(args, profile_values: dict[str, Any], cli_overrides: set[str]) -> None:
    for key, value in profile_values.items():
        if not hasattr(args, key):
            continue
        if key in cli_overrides:
            continue
        setattr(args, key, value)


DEFAULTS_PATH = Path(__file__).parent.parent / "configs" / "defaults.toml"


def build_experiment_config(args, parser) -> ExperimentConfig:
    cli_overrides = _collect_cli_override_keys(parser)

    # Step 1: always apply defaults.toml (base layer)
    if DEFAULTS_PATH.exists():
        default_values = load_profile_values(str(DEFAULTS_PATH))
        _merge_profile_into_namespace(args, default_values, cli_overrides)

    # Step 2: apply named profile from defaults.toml
    if getattr(args, "profile", None):
        profile_values = load_named_profile(args.profile, DEFAULTS_PATH)
        _merge_profile_into_namespace(args, profile_values, cli_overrides)

    if getattr(args, "epochs", None) is not None and getattr(args, "max_steps", None) is not None:
        parser.error("--epochs and --max-steps are mutually exclusive.")

    exp = ExperimentConfig(
        model=args.model,
        run=RunConfig(
            processonly=args.processonly,
            splits_only=args.splits_only,
            cpu=args.cpu,
            dev=args.dev,
            seed=args.seed,
            tag=args.tag,
            skip_fit=args.skip_fit,
            zero_shot=args.zero_shot,
            api_url=getattr(args, "api_url", None),
            api_timeout=getattr(args, "api_timeout", 120),
            reprocess=args.reprocess,
            profile=args.profile,
            opro=args.opro,
            opro_steps=args.opro_steps,
            opro_train_size=args.opro_train_size,
            opro_candidates=args.opro_candidates,
        ),
        data=DataConfig(
            dataset=args.dataset,
            n=args.n,
            batch_size=args.batch_size,
            cw=args.cw,
            date=args.date,
            fold=args.fold,
            total_folds=args.total_folds,
            fold_type=args.fold_type,
            no_holdout=args.no_holdout,
            templated=args.templated,
            mixed_resampling_multiplier=args.mixed_resampling_multiplier,
            mixed_resampling_oversample_cap=args.mixed_resampling_oversample_cap,
            undersampling_multiplier=args.undersampling_multiplier,
            oversampling_multiplier=args.oversampling_multiplier,
            oversampling_cap=args.oversampling_cap,
            cwe_assignment_strategy=args.cwe_assignment_strategy,
            holdout_prop=args.holdout_prop,
            test_prop=args.test_prop,
        ),
        trainer=TrainerConfig(
            grad_acc=args.grad_acc,
            epochs=args.epochs,
            max_steps=getattr(args, "max_steps", None),
            weight_decay=args.weight_decay,
        ),
    )
    exp.validate()
    return exp


def format_resolved_config(config: ExperimentConfig, selected_family: str, checkpoint: str) -> str:
    payload = config.as_resolved_dict()
    payload["selected_family"] = selected_family
    payload["checkpoint"] = checkpoint
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
