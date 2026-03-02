import pathlib

import onnx
import onnxslim
import torch.onnx
from torch import Tensor
from torch.onnx import ONNXProgram

from inference.me_infer import SegmentationEstimationInferenceModel
from lib import logging
from modules.functional import format_boundaries, boundaries_to_regions, regions_to_durations


class WrappedEncoderModel(torch.nn.Module):
    def __init__(self, model: SegmentationEstimationInferenceModel):
        super().__init__()
        self.model = model

    def forward(self, waveform: Tensor, duration: Tensor):
        return self.model.forward_encoder(
            waveform=waveform, duration=duration,
        )


class WrappedSegmenterModel(torch.nn.Module):
    def __init__(self, model: SegmentationEstimationInferenceModel):
        super().__init__()
        self.model = model

    # noinspection PyPep8Naming
    def forward(
            self, x_seg: Tensor, language: Tensor = None, known_boundaries: Tensor = None,
            prev_boundaries: Tensor = None, t: Tensor = None,
            maskT: Tensor = None,
            threshold: Tensor = None, radius: Tensor = None,
    ) -> Tensor:
        return self.model.forward_and_decode_boundaries(
            x_seg=x_seg, known_boundaries=known_boundaries,
            prev_boundaries=prev_boundaries, t=t,
            language=language, mask=maskT,
            threshold=threshold, radius=radius,
        )


class WrappedEstimatorModel(torch.nn.Module):
    def __init__(self, model: SegmentationEstimationInferenceModel):
        super().__init__()
        self.model = model

    def forward(
            self, x_est: Tensor, boundaries: Tensor,
            t_mask: Tensor, n_mask: Tensor,
            threshold: Tensor,
    ) -> tuple[Tensor, Tensor]:
        regions = boundaries_to_regions(boundaries, mask=t_mask)  # [B, T]
        return self.model.forward_and_decode_scores(
            x_est=x_est, regions=regions,
            t_mask=t_mask, n_mask=n_mask,
            threshold=threshold,
        )


class Durations2Boundaries(torch.nn.Module):
    def __init__(self, timestep: float):
        super().__init__()
        self.timestep = timestep

    def forward(self, durations: Tensor, mask: Tensor) -> Tensor:
        boundaries = format_boundaries(
            durations=durations, length=mask.size(1), timestep=self.timestep
        )
        return boundaries


class Boundaries2Durations(torch.nn.Module):
    def __init__(self, timestep: float):
        super().__init__()
        self.timestep = timestep

    def forward(self, boundaries: Tensor, mask: Tensor) -> Tensor:
        regions = boundaries_to_regions(boundaries, mask=mask)
        max_idx = regions.amax(dim=-1, keepdim=True)  # [B, 1]
        N = max_idx.max()
        idx = torch.arange(N, dtype=torch.long, device=regions.device).unsqueeze(0)  # [1, N]
        n_mask = idx < max_idx  # [B, N]
        durations = regions_to_durations(regions, max_n=N) * self.timestep
        return durations, n_mask


class Exporter:
    def __init__(
            self, model: SegmentationEstimationInferenceModel,
            save_dir: str | pathlib.Path,
            opset_version: int = None
    ):
        self.model = model
        if isinstance(save_dir, str):
            save_dir = pathlib.Path(save_dir)
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.opset_version = opset_version
        self.config_path = self.save_dir / "config.json"
        self.encoder_path = self.save_dir / "encoder.onnx"
        self.segmenter_path = self.save_dir / "segmenter.onnx"
        self.estimator_path = self.save_dir / "estimator.onnx"
        self.dur2bd_path = self.save_dir / "dur2bd.onnx"
        self.bd2dur_path = self.save_dir / "bd2dur.onnx"

    def export(self):
        self.export_encoder()
        self.export_segmenter()
        self.export_estimator()
        self.export_converters()
        logging.info(f"Exported encoder to \'{self.encoder_path.as_posix()}\'.")
        logging.info(f"Exported segmenter to \'{self.segmenter_path.as_posix()}\'.")
        logging.info(f"Exported estimator to \'{self.estimator_path.as_posix()}\'.")
        logging.info(f"Exported dur2bd to \'{self.dur2bd_path.as_posix()}\'.")
        logging.info(f"Exported bd2dur to \'{self.bd2dur_path.as_posix()}\'.")

    def export_encoder(self):
        logging.debug("Exporting encoder start.")
        encoder_wrapper = WrappedEncoderModel(self.model)
        program = torch.onnx.export(
            encoder_wrapper,
            (
                torch.randn(4, 44100),
                torch.randn(4),
            ),
            None,
            input_names=[
                "waveform",
                "duration",
            ],
            output_names=[
                "x_seg",
                "x_est",
                "maskT",
            ],
            dynamic_shapes=(
                {
                    0: "B",
                    1: "L",
                },
                {
                    0: "B",
                },
            ),
            opset_version=self.opset_version,
            dynamo=True,
            external_data=False,
            dump_exported_program=False,
        )
        _clear_stacktrace(program)
        program.save(self.encoder_path)
        _slim_onnx_model(self.encoder_path)
        logging.debug("Exporting encoder done.")

    def export_segmenter(self):
        logging.debug("Exporting segmenter start.")
        segmenter_wrapper = WrappedSegmenterModel(self.model)
        example_kwargs = {
            "language": torch.zeros((4,), dtype=torch.int64),
            "known_boundaries": torch.ones(4, 100, dtype=torch.bool),
            "prev_boundaries": torch.ones(4, 100, dtype=torch.bool),
            "t": torch.rand(()),
            "maskT": torch.ones(4, 100, dtype=torch.bool),
            "threshold": torch.tensor(0.5, dtype=torch.float32),
            "radius": torch.tensor(2, dtype=torch.int64),
        }
        dynamic_kwarg_shapes = {
            "language": {
                0: "B",
            },
            "known_boundaries": {
                0: "B",
                1: "T",
            },
            "prev_boundaries": {
                0: "B",
                1: "T",
            },
            "t": {},
            "maskT": {
                0: "B",
                1: "T",
            },
            "threshold": {},
            "radius": {},
        }
        input_kwarg_names = []
        if self.model.model_config.use_languages:
            input_kwarg_names.append("language")
        input_kwarg_names.append("known_boundaries")
        if self.model.model_config.mode == "d3pm":
            input_kwarg_names.append("prev_boundaries")
            input_kwarg_names.append("t")
        input_kwarg_names.extend([
            "maskT",
            "threshold",
            "radius",
        ])
        program = torch.onnx.export(
            segmenter_wrapper,
            torch.randn(4, 100, self.model.model_config.embedding_dim),
            None,
            kwargs={
                k: example_kwargs[k]
                for k in input_kwarg_names
            },
            input_names=[
                "x_seg",
                *input_kwarg_names,
            ],
            output_names=[
                "boundaries",
            ],
            dynamic_shapes={
                "x_seg": {
                    0: "B",
                    1: "T",
                },
                **{
                    k: dynamic_kwarg_shapes[k]
                    for k in input_kwarg_names
                }
            },
            opset_version=self.opset_version,
            dynamo=True,
            external_data=False,
            dump_exported_program=False,
        )
        _clear_stacktrace(program)
        program.save(self.segmenter_path)
        _slim_onnx_model(self.segmenter_path)
        logging.debug("Exporting segmenter done.")

    def export_estimator(self):
        logging.debug("Exporting estimator start.")
        estimator_wrapper = WrappedEstimatorModel(self.model)
        program = torch.onnx.export(
            estimator_wrapper,
            (
                torch.randn(4, 100, self.model.model_config.embedding_dim),
                (torch.arange(0, 100, dtype=torch.int64) % 10 == 0).unsqueeze(0).expand(4, -1),
                torch.ones(4, 100, dtype=torch.bool),
                torch.ones(4, 10, dtype=torch.bool),
                torch.tensor(0.5, dtype=torch.float32),
            ),
            None,
            input_names=[
                "x_est",
                "boundaries",
                "maskT",
                "maskN",
                "threshold",
            ],
            output_names=[
                "presence",
                "scores",
            ],
            dynamic_shapes=(
                {
                    0: "B",
                    1: "T",
                },
                {
                    0: "B",
                    1: "T",
                },
                {
                    0: "B",
                    1: "T",
                },
                {
                    0: "B",
                    1: "N",
                },
                {},
            ),
            opset_version=self.opset_version,
            dynamo=True,
            external_data=False,
            dump_exported_program=False,
        )
        _clear_stacktrace(program)
        program.save(self.estimator_path)
        _slim_onnx_model(self.estimator_path)
        logging.debug("Exporting estimator done.")

    def export_converters(self):
        logging.debug("Exporting dur2bd start.")
        dur2bd = Durations2Boundaries(timestep=self.model.timestep)
        program = torch.onnx.export(
            dur2bd,
            (
                torch.rand(4, 10),
                torch.ones(4, 100, dtype=torch.bool),
            ),
            self.save_dir / "dur2bd.onnx",
            input_names=[
                "durations",
                "maskT",
            ],
            output_names=[
                "boundaries",
            ],
            dynamic_shapes=(
                {
                    0: "B",
                    1: "N",
                },
                {
                    0: "B",
                    1: "T",
                },
            ),
            opset_version=self.opset_version,
            dynamo=True,
            external_data=False,
            dump_exported_program=False,
        )
        _clear_stacktrace(program)
        program.save(self.dur2bd_path)
        _slim_onnx_model(self.dur2bd_path)
        logging.debug("Exporting dur2bd done.")

        logging.debug("Exporting bd2dur start.")
        bd2dur = Boundaries2Durations(timestep=self.model.timestep)
        torch.onnx.export(
            bd2dur,
            (
                (torch.arange(0, 100, dtype=torch.int64) % 10 == 0).unsqueeze(0).expand(4, -1),
                torch.ones(4, 100, dtype=torch.bool),
            ),
            self.bd2dur_path,
            input_names=[
                "boundaries",
                "maskT",
            ],
            output_names=[
                "durations",
                "maskN",
            ],
            dynamic_axes={
                "boundaries": {
                    0: "B",
                    1: "T",
                },
                "maskT": {
                    0: "B",
                    1: "T",
                },
                "durations": {
                    0: "B",
                    1: "N",
                },
                "maskN": {
                    0: "B",
                    1: "N",
                },
            },
            # We don't use Dynamo here because the function contains dynamic shapes depending on inputs.
            opset_version=min(20, self.opset_version),
            dynamo=False,
            external_data=False,
        )
        _slim_onnx_model(self.bd2dur_path)
        logging.debug("Exporting bd2dur done.")


def _clear_stacktrace(program: ONNXProgram):
    for node in program.model.graph.all_nodes():
        node.metadata_props.pop("pkg.torch.onnx.stack_trace", None)


def _slim_onnx_model(path: pathlib.Path):
    model = onnx.load(path)
    slimmed_model = onnxslim.slim(model)
    if slimmed_model is not None:
        onnx.save(slimmed_model, path)
