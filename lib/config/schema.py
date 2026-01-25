import pathlib
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator

from .core import ConfigBaseModel
from .ops import (
    ConfigOperationBase, ConfigOperationContext,
    ref, this, ctx, if_, exists, coalesce
)


class ConfigurationScope:
    SYLLABLES = 0x1
    NOTES = 0x2


class DynamicCheck:
    def __init__(self, expr: ConfigOperationBase, message=None):
        self.expr = expr
        self.message = message

    def run(self, context: ConfigOperationContext):
        if isinstance(self.expr, ConfigOperationBase):
            expr = self.expr.resolve(context)
        else:
            expr = self.expr
        if not expr:
            raise ValueError(
                f"Dynamic check failed.\n"
                f"{'.'.join(str(e) for e in context.current_path)}\n"
                f"  {self.message}"
            )


class RequiredOnGivenScope(DynamicCheck):
    def __init__(self, scope_mask: int):
        super().__init__(
            expr=if_(ctx("scope") & scope_mask, exists(this()), True),
            message="Field required."
        )


class SpectrogramConfig(ConfigBaseModel):
    type: Literal["mel"] = Field("mel")
    num_bins: int = Field(128, gt=0)
    fmin: float = Field(0, ge=0)
    fmax: float = Field(8000, ge=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("binarizer.features.spectrogram.fmin"),
            message="fmax must be greater than fmin."
        )
    })


class BinarizerFeaturesConfig(ConfigBaseModel):
    audio_sample_rate: int = Field(44100, gt=0)
    hop_size: int = Field(441, gt=0)
    fft_size: int = Field(2048, gt=0)
    win_size: int = Field(2048, gt=0)
    spectrogram: SpectrogramConfig = Field(...)


class BinarizerConfig(ConfigBaseModel):
    data_dir: str = Field(...)
    validation_count: int = Field(20, gt=0)
    num_workers: int = Field(0, ge=0)
    features: BinarizerFeaturesConfig = Field(...)

    @property
    def data_dir_resolved(self) -> pathlib.Path:
        return pathlib.Path(self.data_dir).resolve()


class ModelConfig(ConfigBaseModel):
    num_languages: int = Field(127, ge=0)
    in_channels: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features.spectrogram.num_bins")
    })
    backbone_class: str = Field(...)
    backbone_kwargs: dict[str, Any] = Field(...)


class PitchShiftingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.5, ge=0.0, le=1.0)
    min_semitones: float = Field(-12.0)
    max_semitones: float = Field(12.0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.augmentation.pitch_shifting.min_semitones"),
            message="max_semitones must be greater than min_semitones."
        )
    })


class LoudnessScalingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.5, ge=0.0, le=1.0)
    min_db: float = Field(-12.0)
    max_db: float = Field(12.0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.augmentation.loudness_scaling.min_db"),
            message="max_db must be greater than min_db."
        )
    })


class AugmentationConfig(ConfigBaseModel):
    features: BinarizerFeaturesConfig = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features")
    })
    pitch_shifting: PitchShiftingAugmentationConfig = Field(...)
    loudness_scaling: LoudnessScalingAugmentationConfig = Field(...)


class RegionLossConfig(ConfigBaseModel):
    neighborhood_size: int = Field(5, ge=1)
    exponential_decay: bool = Field(False)


class BoundaryLossConfig(ConfigBaseModel):
    bidirectional: bool = Field(False)


class LossConfig(ConfigBaseModel):
    region_loss: RegionLossConfig = Field(...)
    boundary_loss: BoundaryLossConfig = Field(...)


class DataLoaderConfig(ConfigBaseModel):
    max_batch_frames: int = Field(50000, gt=0)
    max_batch_size: int = Field(64, gt=0)
    max_val_batch_frames: int = Field(20000, gt=0)
    max_val_batch_size: int = Field(1, gt=0)
    frame_count_grid: int = Field(6, ge=1)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=0)


class OptimizerConfig(ConfigBaseModel):
    cls: str = Field(...)
    wraps: Literal["parameters", "module"] = Field("parameters")
    kwargs: dict[str, Any] = Field(...)


class LRSchedulerConfig(ConfigBaseModel):
    cls: str = Field(...)
    kwargs: dict[str, Any] = Field(...)
    unit: Literal["step", "epoch"] = Field(...)
    monitor: str | None = Field(None)

    # noinspection PyMethodParameters
    @field_validator("kwargs")
    def check_kwargs(cls, v):
        res = {}
        for key, value in v.items():
            if isinstance(value, dict):
                if "cls" in value:
                    value.setdefault("kwargs", {})
                    value = LRSchedulerConfig.model_validate(value)
                else:
                    value = LRSchedulerConfig.check_kwargs(value)
            elif isinstance(value, list):
                value = [
                    LRSchedulerConfig.model_validate(item) if isinstance(item, dict) and "cls" in item else item
                    for item in value
                ]
            res[key] = value
        return res


class PeriodicCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["periodic"] = Field("periodic")
    unit: Literal["step", "epoch"] = Field(None, json_schema_extra={
        "dynamic_expr": coalesce(this(), ref("training.trainer.unit"))
    })
    since_m_units: int = Field(0, ge=0)
    every_n_units: int = Field(...)
    save_last_k: int = Field(1, ge=-1)
    weights_only: bool = Field(False)


class ExpressionCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["expression"] = Field("expression")
    expression: str = Field(...)
    save_top_k: int = Field(5, ge=-1)
    mode: Literal["max", "min"] = Field(...)
    weights_only: bool = Field(False)


ModelCheckpointConfig = Annotated[
    PeriodicCheckpointConfig | ExpressionCheckpointConfig,
    Field(discriminator="type")
]


class TrainerStrategyConfig(ConfigBaseModel):
    name: str = Field("auto")
    kwargs: dict[str, Any] = Field(...)


class TrainerConfig(ConfigBaseModel):
    unit: Literal["step", "epoch"] = Field(...)
    min_steps: int = Field(0)
    max_steps: int = Field(160000)
    min_epochs: int = Field(0)
    max_epochs: int = Field(1000)
    val_every_n_units: int = Field(..., ge=1)
    log_every_n_steps: int = Field(100, ge=1)
    num_sanity_val_steps: int = Field(1)
    checkpoints: list[ModelCheckpointConfig] = Field(..., min_length=1)
    accelerator: str = Field("auto")
    devices: Union[Literal["auto"], int, list[int]] = Field("auto")
    num_nodes: Literal[1] = Field(1, ge=1)
    strategy: TrainerStrategyConfig = Field(...)
    precision: str = Field("16-mixed")
    accumulate_grad_batches: int = Field(1, ge=1)
    gradient_clip_val: float = Field(1.0, gt=0)

    # noinspection PyMethodParameters
    @field_validator("checkpoints")
    def check_checkpoints(cls, v):
        tags = set()
        for checkpoint in v:
            if checkpoint.tag in tags:
                raise ValueError(f"Duplicate checkpoint tag: '{checkpoint.tag}'.")
            tags.add(checkpoint.tag)
        if all(c.weights_only for c in v):
            raise ValueError("At least one checkpoint should set weights_only to False.")
        return v


class ValidationConfig(ConfigBaseModel):
    spec_vmin: float = Field(-14.0)
    spec_vmax: float = Field(4.0)
    max_plots: int = Field(100, ge=0)


class FinetuningConfig(ConfigBaseModel):
    pretraining_enabled: bool = Field(False)
    pretraining_from: str | None = Field(None, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=if_(ref("training.finetuning.pretraining_enabled"), exists(this()), True),
            message="pretraining_from must be specified if pretraining_enabled is True."
        )
    })
    pretraining_include_params: list[str] = Field(["model.*"])
    pretraining_exclude_params: list[str] = Field([])
    freezing_enabled: bool = Field(False)
    freezing_include_params: list[str] = Field([])
    freezing_exclude_params: list[str] = Field([])


class WeightAveragingConfig(ConfigBaseModel):
    ema_enabled: bool = Field(False)
    ema_decay: float = Field(0.999, gt=0, le=1)
    ema_include_params: list[str] = Field(["model.*"])
    ema_exclude_params: list[str] = Field([])


class TrainingConfig(ConfigBaseModel):
    augmentation: AugmentationConfig = Field(...)
    loss: LossConfig = Field(...)
    dataloader: DataLoaderConfig = Field(...)
    optimizer: OptimizerConfig = Field(...)
    lr_scheduler: LRSchedulerConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    validation: ValidationConfig = Field(...)
    finetuning: FinetuningConfig = Field(...)
    weight_averaging: WeightAveragingConfig = Field(...)


class InferenceConfig(ConfigBaseModel):
    features: BinarizerFeaturesConfig = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features")
    })


class RootConfig(ConfigBaseModel):
    binarizer: BinarizerConfig = Field(...)
    model: ModelConfig = Field(...)
    training: TrainingConfig = Field(...)
    inference: InferenceConfig = Field(...)
