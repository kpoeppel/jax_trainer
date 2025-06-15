# Standard libraries
import importlib
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from glob import glob
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import flax

# JAX/Flax libraries
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml

# from absl import flags, logging
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    dump_config,
    parse_config,
    register,
    register_interface,
)
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.training import train_state
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm

from jax_trainer.callbacks import Callback, ModelCheckpoint
from jax_trainer.datasets import Batch, DatasetModule
from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX
from jax_trainer.logger import (
    HostMetrics,
    ImmutableMetrics,
    LogFreq,
    Logger,
    LogMetricMode,
    LogMode,
    load_pytree,
    save_pytree,
    update_metrics,
)
from jax_trainer.logger.utils import count_parameters, count_parameters_linen
from jax_trainer.nnx_dummy import nnx
from jax_trainer.utils import convert_int_to_prngs

from ..logger.loggers import LoggerConfig
from ..optimizer import OptimizerInterface

LOGGER = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """This is an extension of the optax TrainState to fully cover all states
    of a typical model. Can be extended by custom states.

    This way nnx.BatchNorm and nnx.Dropout are covered.
    """

    graph_def: nnx.GraphDef = None

    batch_stats: nnx.BatchStat = None
    dropout_count: nnx.RngCount = None
    dropout_key: nnx.RngKey = None


class TrainStateLinen(train_state.TrainState):
    """This is an extension of the optax TrainState to fully cover all states
    of a typical model.

    Can be extended by custom states.
    """

    # A simple extension of TrainState to also include mutable variables
    # like batch statistics. If a model has no mutable vars, it is None
    mutable_variables: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


def get_nnx_variable_annotations(cls: train_state.TrainState) -> Dict[str, Any]:
    """Inspect the annotations of a class and return a dictionary of attributes
    that inherit from nnx.Variable.

    Args:
        cls (Type): The class to inspect.

    Returns:
        Dict[str, Any]: A dictionary with attribute names as keys and their annotations as values.
    """
    # Initialize the result dictionary
    nnx_variables = {}

    # Iterate over the class annotations
    for name, annotation in cls.__annotations__.items():
        try:
            # Resolve generic types if any (e.g., Optional, Union)
            origin = getattr(annotation, "__origin__", None) or annotation

            # Check if the annotation is a subclass of nnx.Variable
            if issubclass(origin, nnx.Variable):
                nnx_variables[name] = annotation
        except TypeError:
            # Skip annotations that can't be checked with issubclass
            pass

    return nnx_variables


@dataclass
class TrainerConfig(ConfigInterface):
    logger: LoggerConfig
    seed: int = 42
    seed_eval: int = 43
    callbacks: dict[str, Callback.cfgtype] = field(default_factory=dict)
    train_epochs: int = 1

    check_val_every_n_epoch: int = 1
    # defines a shorter "epoch" without dataset reset at, TODO: implement
    max_steps_per_epoch: int | None = None

    debug: bool = False
    log_param_stats: bool = False
    param_log_modes: Sequence[Literal["min", "max", "mean", "std", "val"]] = (
        "min",
        "max",
        "mean",
        "std",
        "val",
    )
    log_grad_stats: bool = False
    grad_log_modes: Sequence[Literal["min", "max", "mean", "std", "val"]] = (
        "min",
        "max",
        "mean",
        "std",
        "val",
    )
    log_intermediates: bool = False
    log_grad_norm: bool = True
    detect_nans: bool = False
    nan_keys: tuple[str] = ("train/loss",)
    enable_progress_bar: bool = True
    model_mode: Literal["nnx", "linen"] = "nnx"


@register
@register_interface
class TrainerModule(RegistrableConfigInterface):
    rngs: nnx.Rngs | tuple[random.PRNGKey, random.PRNGKey]
    train_state_class = TrainState
    config: TrainerConfig

    def __init__(
        self,
        config: TrainerConfig,
        model_config: BaseModelNNX.cfgtype | BaseModelLinen.cfgtype,
        optimizer_config: OptimizerInterface.cfgtype,
        exmp_input: Batch,
        other_configs: Any = {},
        mesh: jax.sharding.Mesh | None = None,
    ):
        """A basic Trainer module summarizing most common training
        functionalities like logging, model initialization, training loop, etc.

        Args:
            config: A dictionary containing the trainer configuration.
            model_config: A dictionary containing the model configuration.
            optimizer_config: A dictionary containing the optimizer configuration.
            exmp_input: An input to the model with which the shapes are inferred.
        """
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.other_configs = other_configs
        self.exmp_input = exmp_input
        self.mesh = mesh
        self.init_mesh()
        # Initialize RNG first
        self.init_rng()
        self.pre_model_init_callback()
        # Create empty model. Note: no parameters yet
        self.build_model(
            parse_config(BaseModelNNX.cfgtype | BaseModelLinen.cfgtype, model_config), self.rngs
        )
        # Init trainer parts
        self.init_model(exmp_input)
        self.post_model_init_callback()
        self.create_jitted_functions()
        self.init_logger(self.config.logger)
        self.init_callbacks()

    def pre_model_init_callback(self):
        pass

    def post_model_init_callback(self):
        pass

    def init_mesh(self):
        if self.mesh is not None:
            self.model_sharding = NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            self.data_sharding = NamedSharding(self.mesh, jax.sharding.PartitionSpec("data"))

    def state_variables(self):
        """Get the additional state variables (e.g. batch_stats...) excluding
        "graph_def": nnx.GraphDef, "params": nnx.Param,"""
        state_vars = get_nnx_variable_annotations(self.train_state_class)
        return {**state_vars}

    def batch_to_input(self, batch: Batch) -> Any:
        raise NotImplementedError

    def build_model(
        self,
        model_config: BaseModelNNX.cfgtype | BaseModelLinen.cfgtype,
        rngs: nnx.Rngs | tuple[random.PRNGKey, random.PRNGKey],
    ):
        """Creates the model class from the model_config.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        # Create model including parameters except the lazily initialized ones via nnx.bridge.lazy_init
        # This is taken care of in init_model
        if self.config.model_mode == "nnx":
            self.model = model_config.instantiate(BaseModelNNX, rngs=rngs)
        elif self.config.model_mode == "linen":
            self.model = model_config.instantiate(BaseModelLinen)

    def init_logger(self, logger_config: Logger.cfgtype):
        """Initializes a logger and creates a logging directory.

        Args:
            logger_params: A dictionary containing the specification of the logger.
        """
        full_config = {
            "trainer": self.config,
            "model": self.model_config,
            "optimizer": self.optimizer_config,
            **self.other_configs,
        }
        self.logger = Logger(logger_config, dump_config(full_config))
        # Save config and exmp_input
        log_dir = self.logger.log_dir
        LOGGER.info(f"Logging at {log_dir}")
        self.log_dir = log_dir
        self.config.logger.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
        # logging.get_absl_handler().use_absl_log_file(log_dir=log_dir, program_name="absl_logging")
        # logging.set_verbosity(logger_config.get("log_file_verbosity", logging.INFO))
        # logging.set_stderrthreshold(logger_config.get("stderrthreshold", "warning"))
        if not os.path.isfile(os.path.join(log_dir, "config.yaml")):
            config_dict = dump_config(full_config)
            # config_dict = jax.tree_util.tree_map(class_to_name, config_dict)
            with open(os.path.join(log_dir, "config.yaml"), "w") as f:
                yaml.dump(config_dict, f)
        if not os.path.isfile(os.path.join(log_dir, "exmp_input.pkl")):
            inp = self.exmp_input
            inp = jax.device_get(
                jax.experimental.multihost_utils.process_allgather(inp, tiled=True)
            )
            save_pytree(inp, os.path.join(log_dir, "exmp_input.pkl"))
        self.logger.log_scalar(
            "param/num_model_params",
            count_parameters(self.model)
            if self.config.model_mode == "nnx"
            else count_parameters_linen(self.state.params),
            step=0,
        )

    def get_model_rng(self, rng: jax.Array) -> Dict[str, random.PRNGKey]:
        """Returns a dictionary of PRNGKey for init and tabulate.

        Args:
            rng: The current PRNGKey.

        Returns:
            Dict of PRNG Keys
        """
        return {"params": rng}

    def init_callbacks(self):
        """Initializes the callbacks defined in the trainer config."""
        self.callbacks = []
        self.train_step_callbacks = []
        callback_configs = self.config.callbacks
        for name in callback_configs:
            LOGGER.info(f"Initializing callback {name}")
            callback_config = callback_configs[name]
            callback = callback_config.instantiate(Callback, trainer=self, data_module=None)
            self.callbacks.append(callback)
            if hasattr(callback, "on_training_step") and callback.on_training_step is not None:
                self.train_step_callbacks.append(callback)

    def init_rng(self):
        """Initialize the stateful RNG generator."""
        main_rng = random.PRNGKey(self.config.seed)
        if self.config.model_mode == "linen":
            model_rng, init_rng = random.split(main_rng)
            self.rngs = (model_rng, init_rng)
        else:
            self.rngs = nnx.Rngs(main_rng)

    def init_model(self, exmp_input: Batch):
        """Creates an initial training state with newly generated network
        parameters.

        Args:
            exmp_input: An input to the model with which the shapes are inferred.
        """
        # Run model initialization using the stateful RNG
        if self.config.model_mode == "nnx":
            if self.mesh is not None:
                state = nnx.state(self.model)

                sharded_state = jax.lax.with_sharding_constraint(
                    state, nnx.get_named_sharding(state, self.mesh)
                )

                nnx.update(self.model, sharded_state)

                self.run_model_init(exmp_input)
            else:
                self.run_model_init(exmp_input)
        elif self.config.model_mode == "linen":
            # Prepare PRNG and input
            model_rng = random.PRNGKey(self.config.seed)
            model_rng, init_rng = random.split(model_rng)
            # Run model initialization
            variables = self.run_model_init(exmp_input, init_rng=init_rng)
            if isinstance(variables, FrozenDict):
                mutable_variables, params = variables.pop("params")
            else:
                params = variables.pop("params")
                mutable_variables = variables
            if len(mutable_variables) == 0:
                mutable_variables = None
            # Create default state. Optimizer is initialized later
            self.state = TrainStateLinen(
                step=0,
                apply_fn=self.model.apply,
                params=params,
                mutable_variables=mutable_variables,
                rng=model_rng,
                tx=None,
                opt_state=None,
            )

    def init_train_metrics(self, batch: Batch | None = None) -> FrozenDict:
        if self.train_metric_shapes is None:
            if batch is None:
                batch = jax.ShapeDtypeStruct(
                    shape=self.exmp_input.shape,
                    dtype=self.exmp_input.dtype,
                    sharding=self.exmp_input.sharding,
                )
            if self.config.model_mode == "nnx":
                self.train_metric_shapes = nnx.eval_shape(
                    self.train_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    batch=batch,
                    metrics=None,
                )
            elif self.config.model_mode == "linen":
                _, self.train_metric_shapes = jax.eval_shape(
                    self.train_step,
                    state=self.state,
                    batch=batch,
                    metrics=None,
                )
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.train_metric_shapes)

    def init_eval_metrics(self, batch: Batch | None = None) -> FrozenDict:
        if not hasattr(self, "eval_metric_shapes"):
            self.eval_metric_shapes = None
        if self.eval_metric_shapes is None:
            if batch is None:
                batch = jax.ShapeDtypeStruct(
                    shape=self.exmp_input.shape,
                    dtype=self.exmp_input.dtype,
                    sharding=self.exmp_input.sharding,
                )
            if self.config.model_mode == "nnx":
                self.eval_metric_shapes = nnx.eval_shape(
                    self.eval_step, model=self.model, batch=batch, metrics=None
                )
            elif self.config.model_mode == "linen":
                self.eval_metric_shapes = jax.eval_shape(
                    self.eval_step, state=self.state, batch=batch, metrics=None
                )

        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.eval_metric_shapes)

    def set_dataset(self, dataset: DatasetModule):
        for callback in self.callbacks:
            callback.set_dataset(dataset)
        self.dataset = dataset

    def run_model_init(self, exmp_input: Batch, init_rng: jax.Array | None = None) -> Dict:
        """The model initialization call.

        Args:
            exmp_input: An input to the model with which the shapes are inferred.

        Returns:
            The initialized variable dictionary.
        """
        if self.config.model_mode == "nnx":
            exmp_input = self.batch_to_input(exmp_input)
            nnx.bridge.lazy_init(self.model, exmp_input)
            self.model.train()
        elif self.config.model_mode == "linen":
            exmp_input = self.batch_to_input(exmp_input)
            rngs = self.get_model_rng(init_rng)
            variables = self.model.init(rngs, exmp_input)
            if not isinstance(variables, FrozenDict):
                variables = freeze(variables)
            return variables

    def init_optimizer(
        self, num_epochs: int | None = None, num_train_steps_per_epoch: int | None = None
    ):
        """Initializes the optimizer and learning rate scheduler.

        Args:
            num_epochs: Number of epochs the model will be trained for.
            num_train_steps_per_epoch: Number of training steps per epoch.
        """
        optimizer = self.optimizer_config.instantiate(OptimizerInterface)

        if (
            hasattr(optimizer, "scheduler")
            and hasattr(optimizer.scheduler, "config")
            and hasattr(optimizer.scheduler.config, "steps")
            and num_epochs is not None
            and num_train_steps_per_epoch is not None
        ):
            assert (
                optimizer.scheduler.config.steps == num_epochs * num_train_steps_per_epoch
            ), f"Bad step count: {optimizer.scheduler.config.steps} == {num_epochs} * {num_train_steps_per_epoch}"

        if self.config.model_mode == "nnx":
            self.optimizer = nnx.Optimizer(self.model, optimizer)
            if self.mesh is not None:
                state = nnx.state((self.model, self.optimizer))
                sharded_state = jax.lax.with_sharding_constraint(
                    state, nnx.get_named_sharding(state, self.mesh)
                )

                nnx.update((self.model, self.optimizer), sharded_state)

        elif self.config.model_mode == "linen":
            self.state = TrainStateLinen.create(
                apply_fn=self.state.apply_fn,
                params=self.state.params,
                mutable_variables=self.state.mutable_variables,
                tx=optimizer,
                rng=self.state.rng,
            )

    def create_jitted_functions(self):
        """Creates jitted versions of the training and evaluation functions.

        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.config.debug:
            LOGGER.info("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:  # Jit
            train_donate_argnames = ["metrics"]  # Donate metrics to avoid copying.
            if self.config.model_mode == "nnx":
                self.train_step = nnx.jit(
                    train_step,
                    donate_argnames=train_donate_argnames,
                )
                self.eval_step = nnx.jit(
                    eval_step,
                    donate_argnames=["metrics"],  # Donate metrics to avoid copying.
                )
            elif self.config.model_mode == "linen":
                self.train_step = jax.jit(
                    train_step,
                    donate_argnames=train_donate_argnames,
                )
                self.eval_step = jax.jit(
                    eval_step,
                    donate_argnames=["metrics"],  # Donate metrics to avoid copying.
                )

    def loss_function(
        self,
        /,
        model: nnx.Module,
        batch: Batch,
        train: bool = True,
    ) -> Tuple[jnp.array, Dict]:
        """The loss function that is used for training.

        This function needs to be overwritten by a subclass.
        """
        raise NotImplementedError
        # return loss, (mutable_vars, metrics)

    def loss_function_linen(
        self,
        /,
        params: Any,
        state: TrainStateLinen,
        batch: Batch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[jnp.array, Tuple[Any, Dict]]:
        """The loss function that is used for training.

        This function needs to be overwritten by a subclass.
        """
        raise NotImplementedError
        # return loss, metrics

    def model_linen_apply(
        self,
        params: Any,
        state: TrainStateLinen,
        input: Any,
        rng: jax.Array,
        train: bool = True,
        mutable: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Tuple[Any, Dict | None]:
        """The model apply function that can be used in the loss function for
        simplification."""
        rngs = self.get_model_rng(rng)
        variables = {"params": params}
        mutable_keys = [] if mutable is None else mutable
        if state.mutable_variables is not None:
            variables.update(
                {k: state.mutable_variables[k] for k in state.mutable_variables.keys()}
            )
            if train:
                mutable_keys += list(state.mutable_variables.keys())
        if len(mutable_keys) == 0:
            mutable_keys = False
        out = state.apply_fn(
            variables, input, train=train, rngs=rngs, mutable=mutable_keys, **kwargs
        )
        if mutable_keys is not False:
            out, mutable_vars = out
        else:
            mutable_vars = None
        return out, mutable_vars

    def model_apply(
        self,
        model: nnx.Module,
        inputs: Any,
        train: bool = True,
        **kwargs,
    ) -> Tuple[Any, Dict | None]:
        """The model apply function that can be used in the loss function for
        simplification."""

        if train:
            model.train()
        else:
            model.eval()
        out = model(inputs, **kwargs)

        return out

    def extended_inspection(self, model, grads, mutable_vars: Any | None = None):
        step_metrics = {}
        if self.config.log_param_stats:
            if self.config.model_mode == "nnx":
                params = nnx.state(model, nnx.Param)
            elif self.config.model_mode == "linen":
                params = model["params"]
            else:
                raise ValueError
            leaves, _ = jax.tree_util.tree_flatten_with_path(params)
            param_dict = {
                ".".join(
                    filter(
                        lambda x: x is not None,
                        map(lambda x: x.key if hasattr(x, "key") else None, leaf[0]),
                    )
                )
                + "_"
                + op: getattr(jnp, op)(leaf[1])
                for leaf in leaves
                for op in (["min", "max", "mean", "std"] if leaf[1].size > 1 else ["val"])
            }
            param_metrics = {
                k: {"value": v, "count": 1, "log_modes": self.config.params_log_modes}
                for k, v in param_dict.items()
            }
            step_metrics.update(param_metrics)
        if self.config.log_grad_stats:
            leaves, _ = jax.tree_util.tree_flatten_with_path(grads)
            grad_dict = {
                ".".join(
                    filter(
                        lambda x: x is not None,
                        map(lambda x: x.key if hasattr(x, "key") else None, leaf[0]),
                    )
                )
                + ".grad_"
                + op: getattr(jnp, op)(leaf[1])
                for leaf in leaves
                for op in (["min", "max", "mean", "std"] if leaf[1].size > 1 else ["val"])
            }
            grad_metrics = {
                k: {"value": v, "count": 1, "log_modes": self.config.grad_log_modes}
                for k, v in grad_dict.items()
            }
            step_metrics.update(grad_metrics)
        if self.config.log_intermediates:
            if self.config.model_mode == "nnx":
                intermed = nnx.state(model, nnx.Intermediate)
            elif self.config.model_mode == "linen":
                intermed = mutable_vars["intermediates"]
            leaves, _ = jax.tree_util.tree_flatten_with_path(intermed)
            interm_dict = {
                ".".join(
                    filter(
                        lambda x: x is not None,
                        map(lambda x: x.key if hasattr(x, "key") else None, leaf[0]),
                    )
                )
                + "_"
                + op: getattr(jnp, op)(leaf[1])
                for leaf in leaves
                for op in (["min", "max", "mean", "std"] if leaf[1].size > 1 else ["val"])
            }
            interm_metrics = {
                k: {"value": v, "count": 1, "log_modes": self.config.interm_log_modes}
                for k, v in interm_dict.items()
            }
            step_metrics.update(interm_metrics)
        return step_metrics

    def create_training_function(
        self,
    ) -> (
        Callable[
            [TrainStateLinen, Batch, ImmutableMetrics],
            Tuple[TrainStateLinen, ImmutableMetrics],
        ]
        | Callable[[nnx.Module, nnx.Optimizer, Batch, ImmutableMetrics], ImmutableMetrics]
    ):
        """Creates and returns a function for the training step.

        The function takes as input the training state and a batch from
        the train loader. The function is expected to return a
        dictionary of logging metrics, and a new train state.
        """

        if self.config.model_mode == "nnx":

            def train_step(
                model: nnx.Module,
                optimizer: nnx.Optimizer,
                batch: Batch,
                metrics: ImmutableMetrics | None,
            ) -> Tuple[nnx.Module, ImmutableMetrics]:
                loss_fn = partial(self.loss_function, train=True)
                ret, grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
                loss, step_metrics = ret[0], ret[1]
                step_metrics["loss"] = loss
                optimizer.update(grads)
                if self.config.log_grad_norm:
                    grad_norm = optax.global_norm(grads)
                    step_metrics["optimizer/grad_global_norm"] = {
                        "value": grad_norm,
                        "log_freq": LogFreq.STEP,
                    }
                    step_metrics["optimizer/grad_global_norm_max"] = {
                        "value": grad_norm,
                        "mode": LogMetricMode.MAX,
                        "log_freq": LogFreq.EPOCH,
                    }
                    params_norm = optax.global_norm(nnx.state(model, nnx.Param))
                    step_metrics["optimizer/params_global_norm"] = {
                        "value": params_norm,
                        "log_freq": LogFreq.STEP,
                    }

                step_metrics.update(**self.extended_inspection(model, grads))
                metrics = update_metrics(metrics, step_metrics, train=True, batch_size=batch.size)
                return metrics

        else:

            def train_step(
                state: TrainStateLinen, batch: Batch, metrics: ImmutableMetrics | None
            ) -> Tuple[TrainStateLinen, ImmutableMetrics]:
                next_rng, step_rng = random.split(state.rng)

                def loss_fn(params):
                    return self.loss_function_linen(params, state, batch, rng=step_rng, train=True)

                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                loss, mutable_vars, step_metrics = ret[0], *ret[1]
                if mutable_vars is not None:
                    mutable_vars = freeze(
                        mutable_vars
                    )  # Ensure that mutable_vars is a frozen dict.
                step_metrics["loss"] = loss
                state = state.apply_gradients(
                    grads=grads, mutable_variables=mutable_vars, rng=next_rng
                )
                if self.config.log_grad_norm:
                    grad_norm = optax.global_norm(grads)
                    step_metrics["optimizer/grad_global_norm"] = {
                        "value": grad_norm,
                        "log_freq": LogFreq.STEP,
                    }
                    step_metrics["optimizer/grad_global_norm_max"] = {
                        "value": grad_norm,
                        "mode": LogMetricMode.MAX,
                        "log_freq": LogFreq.EPOCH,
                    }
                    params_norm = optax.global_norm(state.params)
                    step_metrics["optimizer/params_global_norm"] = {
                        "value": params_norm,
                        "log_freq": LogFreq.STEP,
                    }
                metrics = update_metrics(metrics, step_metrics, train=True, batch_size=batch.size)
                return state, metrics

        return train_step

    def create_evaluation_function(
        self,
    ) -> (
        Callable[[nnx.Module, Batch, ImmutableMetrics], ImmutableMetrics]
        | Callable[[TrainStateLinen, Batch, ImmutableMetrics | None], ImmutableMetrics]
    ):
        """Creates and returns a function for the evaluation step.

        The function takes as input the training state and a batch from
        the val/test loader. The function is expected to return a
        dictionary of logging metrics, and a new train state.
        """

        if self.config.model_mode == "nnx":

            def eval_step(
                model: nnx.Module, batch: Batch, metrics: ImmutableMetrics | None
            ) -> ImmutableMetrics:
                loss, step_metrics = self.loss_function(model=model, batch=batch, train=False)
                step_metrics["loss"] = loss
                metrics = update_metrics(metrics, step_metrics, train=False, batch_size=batch.size)
                return metrics

        elif self.config.model_mode == "linen":

            def eval_step(
                state: TrainState, batch: Batch, metrics: ImmutableMetrics | None
            ) -> ImmutableMetrics:
                loss, (_, step_metrics) = self.loss_function_linen(
                    params=state.params,
                    state=state,
                    batch=batch,
                    rng=random.PRNGKey(self.config.seed_eval),
                    train=False,
                )
                step_metrics["loss"] = loss
                metrics = update_metrics(metrics, step_metrics, train=False, batch_size=batch.size)
                return metrics

        return eval_step

    def create_functions(
        self,
    ) -> (
        Tuple[
            Callable[[nnx.Module, nnx.Optimizer, Batch, ImmutableMetrics], ImmutableMetrics],
            Callable[[nnx.Module, Batch, ImmutableMetrics], ImmutableMetrics],
        ]
        | Tuple[
            Callable[
                [TrainStateLinen, Batch, ImmutableMetrics],
                Tuple[TrainStateLinen, ImmutableMetrics],
            ],
            Callable[[TrainStateLinen, Batch, ImmutableMetrics | None], ImmutableMetrics],
        ]
    ):
        """Creates and returns functions for the training and evaluation step.

        The functions take as input the training state and a batch from
        the train/ val/test loader. Both functions are expected to
        return a dictionary of logging metrics, and the training
        function a new train state. This function needs to be
        overwritten by a subclass. The train_step and eval_step
        functions here are examples for the signature of the functions.
        """
        return self.create_training_function(), self.create_evaluation_function()

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator | dict[Iterator],
        test_loader: Iterator | dict[Iterator] | None = None,
        num_epochs: int | None = None,
    ) -> Dict[str, Any]:
        """Starts a training loop for the given number of epochs.

        Args:
            train_loader: Data loader of the training set.
            val_loader: Data loader of the validation set.
            test_loader: If given, best model will be evaluated on the test set.
            num_epochs: Number of epochs for which to train the model.

        Returns:
            A dictionary of the train, validation and evt. test metrics for the
            best model on the validation set.
        """
        if num_epochs is None:
            num_epochs = self.config.train_epochs
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        self.global_step = 0
        self.train_metric_shapes = None
        self.eval_metrics_shapes = None
        # Prepare training loop
        self.on_training_start()
        self.test_eval_function(val_loader)
        all_eval_metrics = {}
        train_metrics = None
        training_failed = False

        val_loader = {"val": val_loader} if not isinstance(val_loader, dict) else val_loader
        test_loader = {"test": test_loader} if not isinstance(test_loader, dict) else test_loader

        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            self.on_training_epoch_start(epoch_idx)
            train_metrics, epoch_metrics = self.train_epoch(
                train_loader, epoch_idx=epoch_idx, train_metrics=train_metrics
            )
            if self.config.detect_nans:
                nan_keys = self.config.nan_keys
                if any([np.isnan(epoch_metrics.get(key, 0.0)) for key in nan_keys]):
                    LOGGER.error(
                        f"NaN detected in epoch metrics of epoch {epoch_idx}. Aborting training."
                    )
                    training_failed = True
                    raise ValueError
            self.on_training_epoch_end(epoch_metrics, epoch_idx)
            # Validation every N epochs
            if (
                self.config.check_val_every_n_epoch > 0
                and epoch_idx % self.config.check_val_every_n_epoch == 0
            ):
                self.on_validation_epoch_start(epoch_idx)
                eval_metrics = {}
                for val_idx, (val_name, val_sub_loader) in enumerate(val_loader.items()):
                    eval_metrics.update(
                        **self.eval_model(
                            val_sub_loader,
                            mode="val" if val_idx == 0 else "val_" + val_name,
                            epoch_idx=epoch_idx,
                        )
                    )
                all_eval_metrics[epoch_idx] = eval_metrics
                self.on_validation_epoch_end(eval_metrics, epoch_idx)
        if not training_failed:
            self.on_training_end()
            # Test best model if possible
            if test_loader is not None:
                # self.load_model(raise_if_not_found=False)
                self.on_test_epoch_start(epoch_idx)
                test_metrics = {}
                for test_idx, (test_name, test_sub_loader) in enumerate(test_loader.items()):
                    test_metrics.update(
                        **self.eval_model(
                            test_sub_loader,
                            mode="test" if test_idx == 0 else "test_" + test_name,
                            epoch_idx=epoch_idx,
                        )
                    )
                self.on_test_epoch_end(test_metrics, epoch_idx)
                all_eval_metrics["test"] = test_metrics
        # Close logger
        self.logger.finalize("success" if not training_failed else "failed")
        for callback in self.callbacks:
            callback.finalize("success" if not training_failed else "failed")
        return all_eval_metrics

    def test_model(
        self, test_loader: Iterator, apply_callbacks: bool = False, epoch_idx: int = 0
    ) -> Dict[str, Any]:
        """Tests the model on the given test set.

        Args:
            test_loader: Data loader of the test set.
            apply_callbacks: If True, the callbacks will be applied.
            epoch_idx: The epoch index to use for the callbacks and logging.
        """
        test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch_idx=epoch_idx)
        return test_metrics

    def test_eval_function(self, val_loader: Iterator) -> None:
        """Tests the evaluation function on a single batch.

        This is useful to check if the functions have the correct signature and return the correct
        values. This prevents annoying errors that occur at the first evaluation step.

        This function does not test the training function anymore. This is because the training
        function is already executed in the first epoch and we change its jit signature to donate
        the train state and metrics. Thus, executing a training step requires updating the train
        state, which we would not want to do here. The compilation time is logged during the very
        first training step.

        Args:
            val_loader: Data loader of the validation set.
        """
        print("Verifying evaluation function...")
        val_batch = next(
            iter(
                val_loader if not isinstance(val_loader, dict) else val_loader[list(val_loader)[0]]
            )
        )
        # eval_metrics = self.init_eval_metrics(val_batch)
        start_time = time.time()
        LOGGER.info("Testing and compiling eval_step...")
        if self.config.model_mode == "nnx":
            self.model.eval()
            _ = self.eval_step(self.model, val_batch, metrics=None)  # eval_metrics)

        elif self.config.model_mode == "linen":
            _ = self.eval_step(self.state, val_batch, metrics=None)
        LOGGER.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

    def train_epoch(
        self, train_loader: Iterator, epoch_idx: int, train_metrics: ImmutableMetrics | None
    ) -> Tuple[ImmutableMetrics, HostMetrics]:
        """Trains a model for one epoch.

        Args:
            train_loader: Data loader of the training set.
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the average training metrics over all batches
            for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        LOGGER.info(f"Training epoch... global step: {self.global_step}")
        self.logger.start_epoch(epoch_idx, mode="train")
        if self.config.model_mode == "nnx":
            self.model.train()
        for batch in self.tracker(train_loader, desc="Training", leave=False):
            if train_metrics is None:
                train_metrics = self.init_train_metrics(batch)
            if self.global_step == 0:
                # Log compilation and execution time of the first batch.
                LOGGER.info("Compiling train_step...")
                start_time = time.time()
                if self.config.model_mode == "nnx":
                    train_metrics = self.train_step(
                        self.model, self.optimizer, batch, metrics=train_metrics
                    )
                elif self.config.model_mode == "linen":
                    self.state, train_metrics = self.train_step(
                        self.state, batch, metrics=train_metrics
                    )
                LOGGER.info(
                    f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds."
                )
            else:
                # Annotated with step number for TensorBoard profiling.
                with jax.profiler.StepTraceAnnotation(f"train_step_{self.global_step}"):
                    if self.config.model_mode == "nnx":
                        train_metrics = self.train_step(
                            self.model, self.optimizer, batch, metrics=train_metrics
                        )
                    elif self.config.model_mode == "linen":
                        self.state, train_metrics = self.train_step(
                            self.state, batch, metrics=train_metrics
                        )
            for callback in self.train_step_callbacks:
                callback.on_training_step(train_metrics, epoch_idx, self.global_step)
            train_metrics = self.logger.log_step(train_metrics)
            self.global_step += 1
        train_metrics, epoch_metrics = self.logger.end_epoch(train_metrics)
        return train_metrics, epoch_metrics

    def eval_model(self, data_loader: Iterator, mode: str, epoch_idx: int) -> HostMetrics:
        """Evaluates the model on a dataset.

        Args:
            data_loader: Data loader of the dataset to evaluate on.
            mode: Whether 'val' or 'test'
            epoch_idx: Current epoch index.

        Returns:
            A dictionary of the evaluation metrics, averaged over data points
            in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        self.logger.start_epoch(epoch_idx, mode=mode)
        if self.config.model_mode == "nnx":
            self.model.eval()
        # eval_metrics = self.init_eval_metrics()
        eval_metrics = None
        step_count = 0
        for batch in self.tracker(data_loader, desc=mode.capitalize(), leave=False):
            if self.config.model_mode == "nnx":
                eval_metrics = self.eval_step(self.model, batch, metrics=eval_metrics)
            elif self.config.model_mode == "linen":
                eval_metrics = self.eval_step(self.state, batch, metrics=eval_metrics)
            step_count += 1
        if step_count == 0:
            LOGGER.warning(f"No batches in {mode} loader at epoch {epoch_idx}.")
        _, metrics = self.logger.end_epoch(eval_metrics, save_metrics=True)
        return metrics

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """Wraps an iterator in a progress bar tracker (tqdm) if the progress
        bar is enabled.

        Args:
            iterator: Iterator to wrap in tqdm.
            kwargs: Additional arguments to tqdm.

        Returns:
            Wrapped iterator if progress bar is enabled, otherwise same iterator
            as input.
        """
        if self.config.enable_progress_bar and jax.process_index() == 0:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def on_training_start(self):
        """Method called before training is started.

        Can be used for additional initialization operations etc.
        """
        LOGGER.info("Starting training")
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        """Method called after training has finished.

        Can be used for additional logging or similar.
        """
        LOGGER.info("Finished training")
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self, epoch_idx: int):
        """Method called at the start of each training epoch. Can be used for
        additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch that has started.
        """
        LOGGER.info(f"Starting training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch_idx)

    def on_training_epoch_end(self, train_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each training epoch. Can be used for
        additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch that has finished.
        """
        LOGGER.info(f"Finished training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch_idx)

    def on_validation_epoch_start(self, epoch_idx: int):
        """Method called at the start of each validation epoch. Can be used for
        additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch at which validation was started.
        """
        LOGGER.info(f"Starting validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_start(epoch_idx)

    def on_validation_epoch_end(self, eval_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each validation epoch. Can be used for
        additional logging and evaluation.

        Args:
            epoch_idx: Index of the training epoch at which validation was performed.
            eval_metrics: A dictionary of the validation metrics. New metrics added to
                this dictionary will be logged as well.
            val_loader: Data loader of the validation set, to support additional
                evaluation.
        """
        LOGGER.info(f"Finished validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch_idx)

    def on_test_epoch_start(self, epoch_idx: int):
        """Method called at the start of each test epoch. Can be used for
        additional logging or similar.

        Args:
            epoch_idx: Index of the training epoch at which testing was started.
        """
        LOGGER.info(f"Starting test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_start(epoch_idx)

    def on_test_epoch_end(self, test_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each test epoch. Can be used for
        additional logging and evaluation.

        Args:
            epoch_idx: Index of the training epoch at which testing was performed.
            test_metrics: A dictionary of the test metrics. New metrics added to
                this dictionary will be logged as well.
            test_loader: Data loader of the test set, to support additional
                evaluation.
        """
        LOGGER.info(f"Finished test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_end(test_metrics, epoch_idx)

    def load_model(self, epoch_idx: int = -1, raise_if_not_found: bool = True):
        """Loads model parameters and batch statistics from the logging
        directory."""
        LOGGER.info(f"Loading model from epoch {epoch_idx}")
        state_dict = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_model(epoch_idx)
                break
        if state_dict is None:
            if raise_if_not_found:
                raise ValueError("No model checkpoint callback found in callbacks.")
            else:
                LOGGER.warning("No model checkpoint callback found in callbacks.")
        else:
            self.restore(state_dict)

    def restore(self, state_dict: Dict[str, Any], restore_opt_state: bool = False):
        """Restores the state of the trainer from a state dictionary.

        Args:
            state_dict: State dictionary to restore from.
        """
        LOGGER.info("Restoring trainer state with keys " + str(state_dict.keys()))

        if self.config.model_mode == "nnx":
            state_dict = convert_int_to_prngs(state_dict)

            def flatten_leaf(node):
                if (
                    isinstance(node, dict)
                    and "value" in node
                    and not isinstance(node["value"], dict)
                ):
                    return node["value"]
                return node

            # # TODO: replace this with a more proper check (this causes problems if you have a "value" named layer)
            state_dict = {k: v for k, v in state_dict.items()}
            state_dict = jax.tree.map(
                flatten_leaf,
                state_dict,
                is_leaf=lambda node: isinstance(node, dict)
                and "value" in node
                and not isinstance(node["value"], dict),
            )

            nnx.update(self.model, state_dict["state"])
            if restore_opt_state:
                nnx.update(self.optimizer, state_dict["opt_state"])
                if self.mesh is not None:
                    state = nnx.state((self.model, self.optimizer))
                    sharded_state = jax.lax.with_sharding_constraint(
                        state, nnx.get_named_sharding(state, self.mesh)
                    )
                    nnx.update((self.model, self.optimizer), sharded_state)
            else:
                if self.mesh is not None:
                    state = nnx.state(self.model)
                    sharded_state = jax.lax.with_sharding_constraint(
                        state, nnx.get_named_sharding(state, self.mesh)
                    )
                    nnx.update(self.model, sharded_state)
        elif self.config.model_mode == "linen":
            state_dict.pop("metrics")
            state_dict.pop("metadata")
            if hasattr(self.state, "tx"):
                state_dict["tx"] = self.state.tx if self.state.tx else self.init_optimizer()
            if hasattr(self.state, "opt_state"):
                state_dict["opt_state"] = state_dict.get("opt_state", self.state.opt_state)
            state_dict["rng"] = state_dict.get("rng", self.state.rng)
            state_dict["mutable_variables"] = state_dict.get("mutable_variables", None)
            state_dict["step"] = state_dict.get("step", 0)

            self.state = self.state.__class__(
                apply_fn=self.model.apply,
                # Optimizer will be overwritten when training starts
                **state_dict,
            )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str,
        exmp_input: Batch = None,
        exmp_input_file: str = None,
        batch_size: int = -1,
    ) -> Any:
        """Creates a Trainer object with same hyperparameters and loaded model
        from a checkpoint directory.

        Args:
            checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
            exmp_input: An input to the model for shape inference.

        Returns:
            A Trainer object with model loaded from the checkpoint folder.
        """
        # Load config
        metadata_file = os.path.join(checkpoint, "metadata/metadata")
        assert os.path.isfile(metadata_file), "Could not find metadata file"
        with open(metadata_file, "rb") as f:
            config = json.load(f)
        # Adjust log dir to where its loaded from
        adjusted_checkpoint = checkpoint.split("/")
        if adjusted_checkpoint[-1] == "":
            adjusted_checkpoint = adjusted_checkpoint[:-1]
        if len(adjusted_checkpoint) < 2:
            raise ValueError("Checkpoint path must be at least two levels deep")
        config["trainer"]["logger"]["log_dir"] = os.path.join(*adjusted_checkpoint[:-2])
        # Load example input
        if exmp_input is None:
            if exmp_input_file is None:
                exmp_input_file = os.path.join(checkpoint, "exmp_input.pkl")
            assert os.path.isfile(exmp_input_file), "Could not find example input file"
            exmp_input = load_pytree(exmp_input_file)
        if batch_size > 0:
            exmp_input = exmp_input[:batch_size]

        # Create trainer
        trainer = cls(
            exmp_input=exmp_input,
            config=parse_config(TrainerModule.cfgtype, config["trainer"]),
            model_config=parse_config(
                BaseModelNNX.cfgtype | BaseModelLinen.cfgtype, config["model"]
            ),
            optimizer_config=parse_config(OptimizerInterface.cfgtype, config["optimizer"]),
        )
        # Load model
        trainer.load_model()
        return trainer
