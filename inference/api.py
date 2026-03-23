import json
import pathlib
from typing import Literal

import lightning.pytorch.callbacks
import torch
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_info
from torch import Tensor

from lib import logging
from lib.config.core import ConfigBaseModel
from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import ModelConfig, InferenceConfig, ValidationConfig
from .me_infer import SegmentationEstimationInferenceModel
from .me_infer_module import InferenceModule

__all__ = [
    "load_config_for_inference",
    "load_config_for_evaluation",
    "load_state_dict_for_inference",
    "load_inference_model",
    "infer_model",
]


@rank_zero_only
def _log_config(cfg: ConfigBaseModel):
    formatter = ModelFormatter()
    print(formatter.format(cfg))


def load_config_for_inference(
        path: pathlib.Path,
        scope: int = 0
) -> tuple[ModelConfig, InferenceConfig]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_raw_config(path, inherit=False, overrides=None)
    model_config = ModelConfig.model_validate(config["model"], scope=scope)
    inference_config = InferenceConfig.model_validate(config["inference"], scope=scope)
    model_config.check(scope_mask=scope)
    inference_config.check(scope_mask=scope)

    _log_config(model_config)
    _log_config(inference_config)

    return model_config, inference_config


def load_config_for_evaluation(
        path: pathlib.Path,
        scope: int = 0,
        overrides: list[str] | None = None,
) -> ValidationConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    if overrides:
        overrides = [
            f"training.validation.{override}"
            for override in overrides
        ]
    config = load_raw_config(path, inherit=True, overrides=overrides, subkey="training.validation")
    validation_config = ValidationConfig.model_validate(config, scope=scope)
    validation_config.check(scope_mask=scope)

    _log_config(validation_config)

    return validation_config


def load_state_dict_for_inference(path: pathlib.Path, ema=True) -> dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict: dict = checkpoint.get("state_dict", {})
    if ema and (ema_state_dict := checkpoint.get("ema_state_dict")) is not None:
        state_dict.update(ema_state_dict)
    if not state_dict:
        raise KeyError(f"No valid state dict found in checkpoint: {path}.")
    return state_dict


def load_inference_model(path: pathlib.Path) -> tuple[SegmentationEstimationInferenceModel, dict[str, int] | None]:
    model_config, inference_config = load_config_for_inference(
        path.parent / "config.yaml"
    )
    model = SegmentationEstimationInferenceModel(model_config=model_config, inference_config=inference_config)
    state_dict = load_state_dict_for_inference(path, ema=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model_config.use_languages:
        lang_map_path = path.parent / "lang_map.json"
        if not lang_map_path.exists():
            raise FileNotFoundError(f"Language map file not found for segmentation model: {lang_map_path}")
        with open(lang_map_path, "r") as f:
            lang_map = json.load(f)
    else:
        lang_map = None

    logging.info(f"Loaded model from \'{path}\'.", callback=rank_zero_info)

    return model, lang_map


def infer_model(
        model: SegmentationEstimationInferenceModel,
        dataset: torch.utils.data.Dataset,
        config: ValidationConfig,
        callbacks: list[lightning.pytorch.callbacks.Callback],
        batch_size: int = 1,
        num_workers: int = 0,
        precision: str = "32-true",
        mode: Literal["predict", "evaluate"] = "predict",
):
    module = InferenceModule(model=model, config=config)
    trainer = lightning.pytorch.Trainer(
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        shuffle=False,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate if hasattr(dataset, "collate") else None,
    )
    if mode == "predict":
        trainer.predict(module, dataloader)
    elif mode == "evaluate":
        trainer.test(module, dataloader)
    else:
        raise ValueError(f"Unknown mode: {mode}")
