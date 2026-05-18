#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
import argparse
import warnings
from datetime import timedelta
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch

from src import ROOT_DIR
from src.args import arg_setup
from src.model_definitions import model_definitions, _select_and_configure_family
from src.model_registry import get_model_spec
from src.runtime_config import ExperimentConfig, build_experiment_config, format_resolved_config
from src.utils import MyProgBar, storage_structure


def _setup_loggers(
    checkpoint_dir: Path,
    tag: str,
    dataset: str,
    dataset_class: str,
    fold_type: str,
    date: str,
    fold: int | None,
    total_fold: int,
    model_name: str,
    hyperparameter_string: str,
    cwe_assignment_strategy: str | None = None,
    **kwargs,
) -> tuple[TensorBoardLogger, CSVLogger, ModelCheckpoint, ModelCheckpoint]:
    """Create loggers and checkpoint callbacks for training."""
    logger_name = lambda logger_dir_name: str(
        storage_structure(
            root=ROOT_DIR / "data" / logger_dir_name,
            tag=tag,
            dataset=dataset,
            dataset_class=dataset_class,
            fold_type=fold_type,
            date=date,
            fold=fold,
            total_fold=total_fold,
            model_name=model_name,
            hyperparameter_string=hyperparameter_string,
            cwe_assignment_strategy=cwe_assignment_strategy,
        )
    )
    tb_logger = TensorBoardLogger(
        "data/tb_logs",
        name="/".join(logger_name("tb_logs").split("/")[:-1]),
        version=logger_name("tb_logs").split("/")[-1],
        log_graph=False,
    )
    csv_logger = CSVLogger(
        "data/csv_logs",
        name="/".join(logger_name("csv_logs").split("/")[:-1]),
        version=logger_name("csv_logs").split("/")[-1],
    )
    checkpoint_name = "{epoch:02d}-{step}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        dirpath=str(checkpoint_dir.absolute()),
        filename=checkpoint_name,
        save_top_k=2,
    )
    inprogress_callback = ModelCheckpoint(
        train_time_interval=timedelta(minutes=30),
        dirpath=str(checkpoint_dir.absolute()),
        filename=f"inprogress-" + checkpoint_name,
        save_last=True,
    )
    return tb_logger, csv_logger, checkpoint_callback, inprogress_callback


def _run_opro(
    config: ExperimentConfig,
    model,
    data_module,
    storage_kwargs: dict,
) -> str:
    """Run OPRO prompt optimization. Returns the best prompt template."""
    from src.opro import OPROOptimizer

    data_module.prepare_data()
    data_module.setup()
    opro_dir = storage_structure(
        root=ROOT_DIR / "data" / "checkpoints" / "opro",
        **storage_kwargs,
    )
    optimizer = OPROOptimizer(
        model=model,
        train_data=data_module.val_data,
        steps=config.run.opro_steps,
        train_size=config.run.opro_train_size,
        candidates_per_step=config.run.opro_candidates,
        checkpoint_dir=opro_dir,
        hyperparams={
            "model": config.model,
            "dataset": config.data.dataset,
            "fold_type": config.data.fold_type,
            "fold": config.data.fold,
            "total_folds": config.data.total_folds,
            "seed": config.run.seed,
            "date": config.data.date,
            "tag": config.run.tag,
            "opro_steps": config.run.opro_steps,
            "opro_train_size": config.run.opro_train_size,
            "opro_candidates": config.run.opro_candidates,
            "batch_size": config.data.batch_size,
            "cw": config.data.cw,
        },
    )
    return optimizer.optimize()


def main():
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Main script for the project.")
    parser = arg_setup(parser)
    args = parser.parse_args()
    config = build_experiment_config(args=args, parser=parser)
    if config.run.reprocess:
        warnings.warn("--reprocess is currently informational and does not change behavior yet.", stacklevel=1)
    seed_everything(config.run.seed, workers=True)
    drac_env = os.getenv("SLURM_JOB_ID") is not None

    models = model_definitions(config)

    model_spec = get_model_spec(config.model)
    checkpoint = model_spec.checkpoint

    selected_family = _select_and_configure_family(models, config, model_spec)
    print("Resolved run configuration:")
    print(format_resolved_config(config=config, selected_family=selected_family, checkpoint=checkpoint))

    run_mode_label = "zero_shot" if config.run.zero_shot else "fine_tuning"
    mode_scoped_tag = f"{run_mode_label}/{config.run.tag}"

    # Preprocess dataset if needed
    data_module = models[config.model]["data_module"](**models[config.model]["data_module_args"])
    if config.run.processonly:
        print("Processing dataset only...")
        print("Fold type:", config.data.fold_type)
        if config.run.splits_only:
            print("Splits only: writing split_dataset parquet, skipping tokenization.")
        data_module.prepare_data()
        data_module.setup(splits_only=config.run.splits_only)
        return

    # Load Model
    all_args = {
        **models[config.model]["model_args"],
        **models[config.model]["data_module_args"],
        **models[config.model].get("trainer_args", {}),
    }
    date = models[config.model]["data_module_args"].get("date", "random")
    dataset_size_label = "full" if config.data.n is None else str(config.data.n)
    hyperparameter_string = f"bs{config.data.batch_size}-acc{config.trainer.grad_acc}-cw{config.data.cw}-seed{config.run.seed}-n{dataset_size_label}"
    storage_kwargs = {
        "tag": mode_scoped_tag,
        "dataset": config.data.dataset,
        "dataset_class": data_module.dataset_class.__name__,
        "fold_type": config.data.fold_type,
        "date": date,
        "fold": config.data.fold,
        "total_fold": config.data.total_folds,
        "model_name": config.model,
        "hyperparameter_string": hyperparameter_string,
        "cwe_assignment_strategy": config.data.cwe_assignment_strategy,
    }
    checkpoint_dir: Path = storage_structure(root=ROOT_DIR / "data" / "checkpoints", **storage_kwargs)
    last_model_path: Path = checkpoint_dir / "last.ckpt"
    model = models[config.model]["model"](**all_args)
    trainer_args = models[config.model].get("trainer_args", {})
    os.makedirs(checkpoint_dir, exist_ok=True)

    tb_logger, csv_logger, checkpoint_callback, inprogress_callback = _setup_loggers(
        checkpoint_dir=checkpoint_dir, **storage_kwargs,
    )

    training_callbacks = [
        checkpoint_callback,
        inprogress_callback,
        MyProgBar(leave=False),
    ]
    plugins = []
    if drac_env:
        plugins.append(SLURMEnvironment(auto_requeue=True))

    # OPRO: optimize the user prompt template before evaluation
    opro_tb_logger = None
    opro_csv_logger = None
    if config.run.opro:
        if not config.run.zero_shot:
            raise ValueError("--opro requires --zero-shot.")
        # Use tag (not mode_scoped_tag) for OPRO storage to match original behavior
        opro_storage_kwargs = {**storage_kwargs, "tag": config.run.tag}
        best_prompt = _run_opro(config, model, data_module, opro_storage_kwargs)
        model.user_prompt_template = best_prompt

    active_loggers = [opro_tb_logger, opro_csv_logger] if config.run.opro else [tb_logger, csv_logger]
    trainer = Trainer(
        devices=1,
        accelerator="cuda" if not config.run.cpu else "cpu",
        strategy="auto",
        logger=active_loggers,
        log_every_n_steps=100,
        fast_dev_run=config.run.dev,
        callbacks=training_callbacks,
        deterministic=True,
        plugins=plugins,
        **trainer_args,
    )

    # Fit
    if not config.resolved_skip_fit():
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=last_model_path if last_model_path.exists() else None,
        )

    # Test
    trainer.test(datamodule=data_module, model=model, ckpt_path="best" if checkpoint_callback.best_model_path else None)

    # Holdout Test
    if data_module.holdout_data:
        model.holdout_test = True
        trainer.test(dataloaders=data_module.holdout_dataloader(), model=model, ckpt_path="best" if checkpoint_callback.best_model_path else None)

    for logger in active_loggers:
        logger.finalize("success")
    print("SUCCESS")
