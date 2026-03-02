import json
import pathlib

import torch

from deployment.context import export_mode
from deployment.exporter import Exporter
from inference.me_infer import SegmentationEstimationInferenceModel
from lib import logging


def deploy_model(
        model: SegmentationEstimationInferenceModel,
        lang_map: dict[str, int] | None,
        save_dir: pathlib.Path,
        opset_version: int = None,
):
    with torch.no_grad():
        with export_mode():
            exporter = Exporter(
                model=model,
                save_dir=save_dir,
                opset_version=opset_version,
            )
            exporter.export()
    config = {
        "samplerate": model.inference_config.features.audio_sample_rate,
        "timestep": model.timestep,
        "languages": lang_map,
        "embedding_dim": model.model_config.embedding_dim,
    }
    config_path = save_dir / "config.json"
    with open(config_path, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved config to \'{config_path.as_posix()}\'.")
    logging.success(f"Deployment completed.")
