import datetime
import json
import pathlib
import re
import shutil

from lightning_utilities.core.rank_zero import rank_zero_only

from lib import logging
from lib.config.core import ConfigBaseModel
from lib.config.formatter import format_model
from lib.config.io import load_raw_config, save_raw_config
from lib.config.schema import RootConfig, PeriodicCheckpointConfig, ExpressionCheckpointConfig

__all__ = [
    "load_config_for_training",
    "find_latest_checkpoints",
    "train_model",
]


@rank_zero_only
def _log_config(cfg: RootConfig):
    print(format_model(cfg.model))
    print(format_model(cfg.training))


def load_config_for_training(
        config_path: pathlib.Path,
        scope: int = 0,
        overrides: list[str] = None
) -> RootConfig:
    config = load_raw_config(config_path, inherit=True, overrides=overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    _log_config(config)
    return config


def find_latest_checkpoints(
        ckpt_dir: pathlib.Path,
        candidate_tags: list[str] = None
) -> list[pathlib.Path]:
    candidates = []
    max_step = -1
    for ckpt in ckpt_dir.glob("model-*-steps=*-epochs=*.ckpt"):
        step = int(re.search(r"steps=(\d+)", ckpt.name).group(1))
        if step > max_step:
            max_step = step
            candidates = [ckpt]
        elif step == max_step:
            candidates.append(ckpt)
    for tag in candidate_tags or []:
        filtered_candidates = []
        for ckpt in candidates:
            ckpt_tag = re.search(r"model-(.*?)-steps=", ckpt.name).group(1)
            if tag == ckpt_tag:
                filtered_candidates.append(ckpt)
        if filtered_candidates:
            return filtered_candidates
    return candidates


def train_model(
        config: RootConfig, pl_module_cls,
        ckpt_save_dir: pathlib.Path,
        log_save_dir: pathlib.Path,
        resume_from: pathlib.Path = None
):
    import lightning.pytorch
    import lightning.pytorch.loggers
    from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_info
    from training.pl_module_base import BaseLightningModule

    from training.callbacks import PeriodicModelCheckpoint, ExpressionModelCheckpoint, FriendlyTQDMProgressBar
    from training.strategy import get_strategy

    if not issubclass(pl_module_cls, BaseLightningModule):
        raise ValueError(f"pl_module_cls must be a subclass of {BaseLightningModule.__name__}")
    logging.info(f"Lightning module: {pl_module_cls.__name__}.", callback=rank_zero_info)

    @rank_zero_only
    def _check_file_and_config(file: pathlib.Path, cfg: ConfigBaseModel):
        cfg_load = load_raw_config(file, inherit=False, overrides=None)
        if cfg_load != cfg.model_dump():
            raise RuntimeError(
                f"Contents of '{file}' do not match the configuration. "
                f"If you edited the configuration file, please re-binarize the dataset."
            )

    @rank_zero_only
    def _check_and_copy(filename: str, from_dir: pathlib.Path, to_dir: pathlib.Path):
        source_file = from_dir / filename
        target_file = to_dir / filename
        if target_file.exists():
            with (
                open(source_file, "r", encoding="utf8") as f1,
                open(target_file, "r", encoding="utf8") as f2,
            ):
                json1 = json.load(f1)
                json2 = json.load(f2)
            if json1 != json2:
                raise RuntimeError(
                    f"Contents of '{source_file}' and '{target_file}' are not identical. "
                    f"If you edited the configuration file, please re-binarize the dataset."
                )
        else:
            shutil.copy(source_file, to_dir)

    @rank_zero_only
    def _config_dump(cfg: RootConfig, to_dir: pathlib.Path):
        # config for inference and exporting
        save_raw_config(cfg.model_dump(include={"model", "inference"}), to_dir / "config.yaml")
        # config for debugging, add timestamp to avoid overwriting
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_raw_config(cfg.model_dump(), to_dir / f"hparams-{current_time}.yaml")

    data_dir = config.binarizer.data_dir_resolved
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    _check_file_and_config(data_dir / "feature.yaml", config.binarizer.features)
    _check_and_copy("lang_map.json", data_dir, ckpt_save_dir)
    _config_dump(config, ckpt_save_dir)
    model_config = config.model
    training_config = config.training

    pl_module: BaseLightningModule = pl_module_cls(
        data_dir=data_dir,
        model_config=model_config,
        training_config=training_config,
        load_pretrained=resume_from is None,
    )
    rank_zero_info(f"Architecture: {pl_module}")
    if resume_from is None:
        logging.info(
            f"No checkpoint found or specified to resume from. Starting new training.",
            callback=rank_zero_info
        )
    else:
        logging.info(f"Resuming training from checkpoint: '{resume_from}'.", callback=rank_zero_info)

    if training_config.trainer.unit == "step":
        val_check_interval = (
                training_config.trainer.val_every_n_units * training_config.trainer.accumulate_grad_batches
        )
        check_val_every_n_epoch = None
    elif training_config.trainer.unit == "epoch":
        val_check_interval = None
        check_val_every_n_epoch = training_config.trainer.val_every_n_units
    else:
        raise ValueError(f"Unit must be 'step' or 'epoch', got '{training_config.trainer.unit}'.")
    callbacks = [
        FriendlyTQDMProgressBar()
    ]
    for ckpt_config in training_config.trainer.checkpoints:
        if ckpt_config.type == "periodic":
            ckpt_config: PeriodicCheckpointConfig
            checkpoint = PeriodicModelCheckpoint(
                dirpath=ckpt_save_dir,
                tag=ckpt_config.tag,
                unit=ckpt_config.unit,
                every_n_units=ckpt_config.every_n_units,
                since_m_units=ckpt_config.since_m_units,
                save_last_k=ckpt_config.save_last_k,
                save_weights_only=ckpt_config.weights_only,
            )
        elif ckpt_config.type == "expression":
            ckpt_config: ExpressionCheckpointConfig
            checkpoint = ExpressionModelCheckpoint(
                dirpath=ckpt_save_dir,
                tag=ckpt_config.tag,
                expression=ckpt_config.expression,
                mode=ckpt_config.mode,
                save_top_k=ckpt_config.save_top_k,
                save_weights_only=ckpt_config.weights_only,
            )
        else:
            raise ValueError(f"Invalid checkpoint monitor type: {ckpt_config.type}")
        callbacks.append(checkpoint)
    trainer = lightning.pytorch.Trainer(
        accelerator=training_config.trainer.accelerator,
        # TODO: strategy
        strategy=get_strategy(
            devices=training_config.trainer.devices,
            num_nodes=training_config.trainer.num_nodes,
            accelerator=training_config.trainer.accelerator,
            strategy={
                "name": training_config.trainer.strategy.name,
                **training_config.trainer.strategy.kwargs,
            },
            precision=training_config.trainer.precision,
        ),
        devices=training_config.trainer.devices,
        num_nodes=training_config.trainer.num_nodes,
        precision=training_config.trainer.precision,
        logger=lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=log_save_dir,
            name="lightning_logs",
            version="latest",
        ),
        callbacks=callbacks,
        min_steps=training_config.trainer.min_steps,
        max_steps=training_config.trainer.max_steps,
        min_epochs=training_config.trainer.min_epochs,
        max_epochs=training_config.trainer.max_epochs,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=training_config.trainer.num_sanity_val_steps,
        log_every_n_steps=1,
        accumulate_grad_batches=training_config.trainer.accumulate_grad_batches,
        gradient_clip_val=training_config.trainer.gradient_clip_val,
        use_distributed_sampler=False,
    )
    trainer.fit(model=pl_module, ckpt_path=resume_from)
