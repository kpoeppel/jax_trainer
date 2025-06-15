import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp
from absl import logging
from compoconf import dump_config, register
from packaging import version

from jax_trainer.callbacks.callback import Callback, CallbackConfig
from jax_trainer.nnx_dummy import nnx
from jax_trainer.utils import convert_prngs_to_int

# ----------------------------------------------------------------------
#    Version check helper
# ----------------------------------------------------------------------
_ORBAX_VERSION = version.parse(getattr(ocp, "__version__", "0.0.0"))
_USE_NEW_API = _ORBAX_VERSION >= version.parse("0.5.0")


@dataclass
class ModelCheckpointConfig(CallbackConfig):
    monitor: str = "val/loss"
    save_optimizer_state: bool = True
    save_top_k: int = 2
    mode: Literal["min", "max"] = "min"


@register
class ModelCheckpoint(Callback):
    """Callback to save model parameters and mutable variables, on old or new
    Orbax."""

    config: ModelCheckpointConfig

    def __init__(self, config, trainer, data_module=None):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Common options
        options = ocp.CheckpointManagerOptions(
            max_to_keep=config.save_top_k,
            best_fn=lambda m: m[config.monitor] if config.monitor in m else 0.0,
            best_mode=config.mode,
            save_interval_steps=1,
            step_prefix="checkpoint",
            cleanup_tmp_directories=True,
            create=True,
        )
        # metadata dumped as JSON-text
        self.metadata = dump_config(
            {
                "trainer": self.trainer.config.to_dict(),
                "model": self.trainer.model_config.to_dict(),
                "optimizer": self.trainer.optimizer_config.to_dict(),
            }
        )

        # build the item_handlers mapping once
        h: Dict[str, ocp.CheckpointHandler] = {}
        if self.trainer.config.model_mode == "nnx":
            h["state"] = ocp.PyTreeCheckpointHandler()
            h["metadata"] = ocp.JsonCheckpointHandler()
            if config.save_optimizer_state:
                h["opt_state"] = ocp.PyTreeCheckpointHandler()
        else:  # linen
            h["step"] = ocp.ArrayCheckpointHandler()
            h["params"] = ocp.StandardCheckpointHandler()
            h["rng"] = ocp.ArrayCheckpointHandler()
            h["metadata"] = ocp.JsonCheckpointHandler()
            if self.trainer.state.mutable_variables is not None:
                h["mutable_variables"] = ocp.StandardCheckpointHandler()
            if config.save_optimizer_state:
                h["optimizer"] = ocp.PyTreeCheckpointHandler()

        # instantiate manager via old or new API
        if _USE_NEW_API:
            # new: pass item_names & item_handlers
            self.manager = ocp.CheckpointManager(
                directory=os.path.abspath(ckpt_dir),
                options=options,
                item_names=tuple(h.keys()),
                item_handlers=h,
            )
        else:
            # legacy: only directory + a single checkpointer;
            # we’ll bundle everything under one PyTreeCheckpointer
            # and pass items/save_kwargs at save time

            # legacy_checkpointer = CompositeCheckpointHandler(h.values())
            self.manager = ocp.CheckpointManager(
                os.path.abspath(ckpt_dir),
                {h_key: ocp.Checkpointer(h_val) for h_key, h_val in h.items()},
                options,
            )
            # store legacy handlers so we can split items/save_kwargs later
            self._legacy_item_handlers = h

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx):
        self.save_model(eval_metrics, epoch_idx)

    def save_model(self, eval_metrics, epoch_idx):
        logging.info(f"Saving model at epoch {epoch_idx} with metrics {eval_metrics}.")
        assert (
            self.config.monitor in eval_metrics
        ), f'Metric "{self.config.monitor}" missing; got {list(eval_metrics)}'

        # collect actual payload
        if self.trainer.config.model_mode == "nnx":
            state = nnx.state(self.trainer.model)
            state = convert_prngs_to_int(state)
            state = jax.tree_map(lambda x: np.array(x), state)
            payload = {
                "state": state,
                "metadata": self.metadata,
                **(
                    {
                        "opt_state": jax.tree_map(
                            lambda x: np.array(x), nnx.state(self.trainer.optimizer)
                        )
                    }
                    if self.config.save_optimizer_state
                    else {}
                ),
            }
        else:
            payload = {
                "step": self.trainer.state.step,
                "params": self.trainer.state.params,
                "rng": self.trainer.state.rng,
                "metadata": self.metadata,
                **(
                    {"mutable_variables": self.trainer.state.mutable_variables}
                    if self.trainer.state.mutable_variables is not None
                    else {}
                ),
                **(
                    {"optimizer": self.trainer.state.optimizer}
                    if self.config.save_optimizer_state
                    else {}
                ),
            }

        # filter metrics to simple types
        metrics = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float, str, bool))}

        if _USE_NEW_API:
            # new-API: wrap each with an ocp.args.* and Composite
            from orbax.checkpoint import args

            save_args = {}
            for name, obj in payload.items():
                if name in ("state", "opt_state"):
                    save_args[name] = args.PyTreeSave(obj)
                elif name in ("step", "rng"):
                    save_args[name] = args.ArraySave(obj)
                elif name in ("params", "mutable_variables", "optimizer"):
                    save_args[name] = args.StandardSave(obj)
                elif name == "metadata":
                    save_args[name] = args.JsonSave(obj)
            composite = args.Composite(**save_args)
            ok = self.manager.save(epoch_idx, args=composite, metrics=metrics, force=True)

        else:
            # legacy: pass items & save_kwargs based on our stored handlers
            items = payload
            save_kwargs = {}
            # figure out per-item arguments (here: empty, but you could pass save_args
            # through self._legacy_item_handlers if you wanted finer control)
            for name, handler in self._legacy_item_handlers.items():
                save_kwargs[name] = {}
            ok = self.manager.save(
                epoch_idx, items=items, save_kwargs=save_kwargs, metrics=metrics, force=True
            )

        assert ok, "Could not save model checkpoint."

    def load_model(self, epoch_idx=-1):
        logging.info(f"Loading model at epoch {epoch_idx}.")
        if epoch_idx == -1:
            epoch_idx = self.manager.best_step() or 0

        if _USE_NEW_API:
            # new: restore returns the Composite-like dict
            restored = self.manager.restore(epoch_idx)
        else:
            # legacy: manager.restore(step) returns the raw items dict
            restored = self.manager.restore(epoch_idx)

        if self.trainer.config.model_mode == "linen":
            # nothing special: just return dict of arrays
            return dict(restored)
        else:
            return restored

    def finalize(self, status: Optional[str] = None):
        logging.info("Closing checkpoint manager")
        self.manager.wait_until_finished()
        self.manager.close()
