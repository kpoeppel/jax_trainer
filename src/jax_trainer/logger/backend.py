import json
import logging
import os
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .utils import flatten_dict

LOGGER = logging.getLogger(__name__)

# from jax_trainer.utils import class_to_name


# def get_logging_dir(logger_config: ConfigDict, full_config: ConfigDict):
#     """Returns the logging directory and version.

#     Args:
#         logger_config (ConfigDict): The logger config.
#         full_config (ConfigDict): The full config of the trainer. Used for getting the model name for the default logging directory.

#     Returns:
#         Tuple[str, str]: The logging directory and version.
#     """
#     # Determine logging directory
#     log_dir = logger_config.get("log_dir", None)
#     if log_dir == "None":
#         log_dir = None
#     if not log_dir:
#         base_log_dir = logger_config.get("base_log_dir", "checkpoints/")
#         # Prepare logging
#         if "model_log_dir" not in logger_config:
#             model_name = full_config.model.name
#             if not isinstance(model_name, str):
#                 model_name = model_name.__name__
#             model_name = model_name.split(".")[-1]
#         else:
#             model_name = logger_config.model_log_dir
#         log_dir = os.path.join(base_log_dir, model_name)
#         if logger_config.get("logger_name", None) is not None and logger_config.logger_name != "":
#             log_dir = os.path.join(log_dir, logger_config.logger_name)
#             version = ""
#         else:
#             version = None
#     else:
#         version = ""
#     return log_dir, version


@register_interface
class ToolLoggerInterface(RegistrableConfigInterface):
    def __init__(self, config: Any, full_config: dict):
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int):
        raise NotImplementedError

    @abstractmethod
    def finalize(self, status: str):
        raise NotImplementedError

    @abstractmethod
    def log_embedding(
        self, key: str, encodings: np.ndarray, metadata: Any, label_img: Any, step: int
    ):
        raise NotImplementedError

    @abstractmethod
    def log_figure(
        self,
        key: str,
        figure: plt.Figure,
        step: int | None = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def log_image(
        self,
        key: str,
        image: np.ndarray,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def save_metrics(self, filename: str, metrics: dict[str, Any]):
        raise NotImplementedError


@dataclass
class WandbConfig(ConfigInterface):
    save_dir: str = ""
    version: str = ""
    name: str | None = None
    project: str | None = None
    entity: str | None = None
    use_timestamp_version: bool = True

    def __post_init__(self):
        if not self.version and self.use_timestamp_version:
            self.version = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


@dataclass
class TensorboardConfig(ConfigInterface):
    save_dir: str = ""
    name: str = ""
    version: str = ""

    use_timestamp_version: bool = True

    def __post_init__(self):
        if not self.version and self.use_timestamp_version:
            self.version = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


@register
class TensorboardToolLogger(ToolLoggerInterface):
    config: TensorboardConfig

    def __init__(self, config: TensorboardConfig, full_config: dict):
        self.config = config
        kwargs = asdict(config)
        del kwargs["use_timestamp_version"]
        del kwargs["class_name"]
        self.logger = TensorBoardLogger(**kwargs)
        self.logger.log_hyperparams(flatten_dict(full_config))

    def finalize(self, status: str):
        self.logger.finalize(status)

    def log_metrics(self, metrics: dict[str, Any], step: int):
        self.logger.log_metrics(metrics, step=step)

    def log_embedding(
        self,
        key: str,
        encodings: np.ndarray,
        metadata: Any,
        label_img: Any,
        logging_mode: str,
        log_postfix: str,
        step: int,
    ):
        import torch

        label_img = np.transpose(label_img, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        label_img = torch.from_numpy(label_img)
        self.logger.experiment.add_embedding(
            tag=f"{logging_mode}/{key}{log_postfix}",
            mat=encodings,
            metadata=metadata,
            label_img=label_img,
            global_step=step,
        )

    def log_figure(
        self,
        key: str,
        figure: plt.Figure,
        step: int | None = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        self.logger.experiment.add_figure(
            tag=f"{logging_mode}/{key}{log_postfix}", figure=figure, global_step=step
        )

    def log_image(
        self,
        key: str,
        image: np.ndarray,
        step: int | None = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        self.logger.experiment.add_image(
            tag=f"{logging_mode}/{key}{log_postfix}",
            img_tensor=image,
            global_step=step,
            dataformats="HWC",
        )

    def save_metrics(self, filename: str, metrics: dict[str, Any]):
        metrics = {
            k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float, str, bool))
        }
        with open(os.path.join(self.logger.save_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)


@register
class WandbToolLogger(ToolLoggerInterface):
    config: WandbConfig

    def __init__(self, config: WandbConfig, full_config: dict):
        self.config = config
        kwargs = asdict(config)
        del kwargs["use_timestamp_version"]
        del kwargs["class_name"]

        self.logger = WandbLogger(config=full_config, log_model=False, **kwargs)

    def finalize(self, status: str):
        self.logger.finalize(status)

    def log_metrics(self, metrics: Any, step: int):
        self.logger.log_metrics(metrics, step=step)

    def log_embedding(
        self, key: str, encodings: np.ndarray, metadata: Any, label_img: Any, step: int
    ):
        LOGGER.warning("Embedding logging not implemented for Weights and Biases.")

    def log_figure(
        self,
        key: str,
        figure: plt.Figure,
        step: int | None = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        self.logger.experiment.log({f"{logging_mode}/{key}{log_postfix}": figure}, step=step)

    def log_image(
        self,
        key: str,
        image: np.ndarray,
        log_postfix: str = "",
        logging_mode: str | None = None,
        step: int | None = None,
    ):
        self.logger.log_image(key=f"{logging_mode}/{key}{log_postfix}", images=[image], step=step)

    def save_metrics(self, filename: str, metrics: dict[str, Any]):
        metrics = {
            k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float, str, bool))
        }
        with open(os.path.join(self.logger.save_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)
