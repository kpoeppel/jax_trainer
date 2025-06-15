from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging
from compoconf import ConfigInterface, register

from jax_trainer.callbacks.callback import Callback, CallbackConfig


@dataclass
class ConfusionMatrixConfig(CallbackConfig):
    normalize: bool = False
    format: str | None = None
    fig_size: tuple[float, float] = (8, 8)
    cmap: str = "Blues"
    dpi: int = 100

    def __post_init__(self):
        if self.normalize:
            self.format = ".2%"
        else:
            self.format = "d"


@register
class ConfusionMatrixCallback(Callback):
    """Callback to visualize the confusion matrix."""

    config: ConfusionMatrixConfig

    def __init__(self, config: ConfusionMatrixConfig, trainer, data_module=None):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx):
        return self._visualize_confusion_matrix(eval_metrics, epoch_idx)

    def _on_test_epoch_end(self, test_metrics, epoch_idx):
        return self._visualize_confusion_matrix(test_metrics, epoch_idx)

    def _visualize_confusion_matrix(self, metrics, epoch_idx):
        conf_key = [k for k in metrics.keys() if k.endswith("conf_matrix")]
        if len(conf_key) == 0:
            logging.warning(
                f"Confusion matrix not found in eval metrics, only found {metrics.keys()}."
            )
            return
        conf_key = conf_key[0]
        conf_matrix = metrics[conf_key]
        if self.config.normalize:
            conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        sns.heatmap(conf_matrix, annot=True, cmap=self.config.cmap, ax=ax, fmt=self.config.format)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion matrix")
        ax.set_xticks(np.arange(conf_matrix.shape[0]) + 0.5)
        ax.set_yticks(np.arange(conf_matrix.shape[1]) + 0.5)
        if self.data_module is not None and hasattr(self.data_module.train_dataset, "class_names"):
            ax.set_xticklabels(self.data_module.train_dataset.class_names)
            ax.set_yticklabels(self.data_module.train_dataset.class_names)
        else:
            ax.set_xticklabels(range(conf_matrix.shape[0]))
            ax.set_yticklabels(range(conf_matrix.shape[1]))
        fig.tight_layout()
        self.trainer.logger.log_figure("confusion_matrix", fig, epoch_idx)
