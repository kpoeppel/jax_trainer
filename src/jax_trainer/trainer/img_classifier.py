from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

import jax
import jax.numpy as jnp
import optax
from compoconf import parse_config, register
from flax import linen as nn
from jax import random

from jax_trainer.datasets import SupervisedBatch
from jax_trainer.interfaces import (
    BaseAugmentationLinen,
    BaseAugmentationNNX,
    BaseAugmentationNNXConfig,
    BasePreprocessingLinen,
    BasePreprocessingNNX,
    BasePreprocessingNNXConfig,
    PyTree,
)
from jax_trainer.logger import LogFreq, LogMetricMode, LogMode, Metrics
from jax_trainer.nnx_dummy import nnx
from jax_trainer.trainer.trainer import TrainerConfig, TrainerModule, TrainState


@dataclass
class ImgClassifierTrainerConfig(TrainerConfig):
    num_classes: int = -1

    def __post_init__(self):
        assert self.num_classes > 1


@register
class ImgClassifierTrainer(TrainerModule):
    config: ImgClassifierTrainerConfig

    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.input

    def loss_function_linen(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict] | Dict]:
        """Loss function for image classification.

        Args:
            params: Parameters of the model.
            state: State of the trainer.
            batch: Batch of data. Assumes structure of SupervisedBatch or subclasses.
            rng: Key for random number generation.
            train: Whether the model is in training mode.

        Returns:
            Tuple of loss and tuple of mutable variables and metrics.
        """
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_linen_apply(
            params=params, state=state, input=imgs, rng=rng, train=train
        )

        if jnp.issubdtype(labels.dtype, jnp.integer):
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            preds = logits.argmax(axis=-1)
            acc = (preds == labels).mean()
        else:
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            preds = logits.argmax(axis=-1)
            acc = (jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1).mean()
        # conf_matrix = jnp.zeros((logits.shape[-1], logits.shape[-1]))
        # conf_matrix = conf_matrix.at[preds, labels].add(1)
        metrics = {
            "acc": acc,
            "acc_std": {"value": acc, "mode": LogMetricMode.STD, "log_mode": LogMode.EVAL},
            "acc_max": {
                "value": acc,
                "mode": LogMetricMode.MAX,
                "log_mode": LogMode.TRAIN,
                "log_freq": LogFreq.EPOCH,
            },
            # "conf_matrix": {
            #     "value": conf_matrix,
            #     "mode": LogMetricMode.SUM,
            #     "log_mode": LogMode.EVAL,
            # },
        }
        return loss, (mutable_variables, metrics)

    def loss_function(
        self,
        model: nnx.Module,
        batch: SupervisedBatch,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict] | Dict]:
        """Loss function for image classification.

        Args:
            params: Parameters of the model.
            state: State of the trainer.
            batch: Batch of data. Assumes structure of SupervisedBatch or subclasses.
            rng: Key for random number generation.
            train: Whether the model is in training mode.

        Returns:
            Tuple of loss and tuple of mutable variables and metrics.
        """
        imgs = batch.input
        labels = batch.target
        logits = self.model_apply(
            model=model,
            inputs=imgs,
            train=train,
        )

        if jnp.issubdtype(labels.dtype, jnp.integer):
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            preds = logits.argmax(axis=-1)
            acc = (preds == labels).mean()
        else:
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            preds = logits.argmax(axis=-1)
            acc = (jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1).mean()
        # conf_matrix = jnp.zeros((logits.shape[-1], logits.shape[-1]))
        # conf_matrix = conf_matrix.at[preds, labels].add(1)
        metrics = {
            "acc": acc,
            "acc_std": {"value": acc, "mode": LogMetricMode.STD, "log_mode": LogMode.EVAL},
            "acc_max": {
                "value": acc,
                "mode": LogMetricMode.MAX,
                "log_mode": LogMode.TRAIN,
                "log_freq": LogFreq.EPOCH,
            },
            # "conf_matrix": {
            #     "value": conf_matrix,
            #     "mode": LogMetricMode.SUM,
            #     "log_mode": LogMode.EVAL,
            # },
        }
        return loss, metrics


class AugmentationTrainState(TrainState):
    augmentation_count: nnx.RngCount = None
    augmentation_key: nnx.RngKey = None


@dataclass
class AugmentedImgClassifierTrainerConfig(TrainerConfig):
    augmentation_seed: int = 44
    num_classes: int = -1
    augmentation: BaseAugmentationLinen.cfgtype | BaseAugmentationNNX.cfgtype = field(
        default_factory=BaseAugmentationNNXConfig
    )
    preprocessing: BasePreprocessingNNX.cfgtype | BasePreprocessingLinen.cfgtype = field(
        default_factory=BasePreprocessingNNXConfig
    )
    loss_type: Literal["cross_entropy", "binary_cross_entropy"] = "cross_entropy"

    def __post_init__(self):
        assert self.num_classes > 1, "Num of classes have to be set for trainer"


@register
class AugmentedImgClassifierTrainer(TrainerModule):
    config: AugmentedImgClassifierTrainerConfig
    train_state_class = AugmentationTrainState

    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        if self.config.model_mode == "nnx":
            imgs, labels = self.preprocessing.preprocess((batch.input, batch.target))
        else:
            imgs, labels = self.preprocessing.apply(
                {}, (batch.input, batch.target), method=self.preprocessing.preprocess
            )
        return imgs

    def init_rng(self):
        """Initialize the stateful RNG generator."""
        main_rng = random.PRNGKey(self.config.seed)
        if self.config.model_mode == "linen":
            model_rng, init_rng = random.split(main_rng)
            self.rngs = (model_rng, init_rng)
        else:
            self.rngs = nnx.Rngs(main_rng, augmentation=self.config.augmentation_seed)

    def pre_model_init_callback(self):
        self.build_preprocessing()

    def post_model_init_callback(self):
        self.build_augmentation()

    def build_augmentation(self):
        if self.config.model_mode == "nnx":
            self.augmentation_config = parse_config(
                BaseAugmentationLinen.cfgtype | BaseAugmentationNNX.cfgtype,
                self.config.augmentation,
            )
            augmented_model = self.augmentation_config.instantiate(
                BaseAugmentationNNX,
                wrapped_model=self.model,
                rngs=self.rngs,
                mesh=self.mesh,
            )
        elif self.config.model_mode == "linen":
            self.augmentation_config = parse_config(
                BaseAugmentationLinen.cfgtype | BaseAugmentationNNX.cfgtype,
                self.config.augmentation,
            )

            augmented_model = self.augmentation_config.instantiate(
                BaseAugmentationLinen, self.model, mesh=self.mesh
            )
        self.model = augmented_model

    def build_preprocessing(self):
        if self.config.model_mode == "nnx":
            self.preprocessing_config = parse_config(
                BasePreprocessingLinen.cfgtype | BasePreprocessingNNX.cfgtype | None,
                self.config.preprocessing,
            )
            self.preprocessing = self.preprocessing_config.instantiate(
                BasePreprocessingNNX, mesh=self.mesh
            )

        elif self.config.model_mode == "linen":
            self.preprocessing_config = parse_config(
                BasePreprocessingLinen.cfgtype | BasePreprocessingNNX.cfgtype | None,
                self.config.preprocessing,
            )
            self.preprocessing = self.preprocessing_config.instantiate(
                BasePreprocessingLinen, mesh=self.mesh
            )
            self.preprocessing.init(jax.random.PRNGKey(0), self.exmp_input)

    def model_augment(
        self,
        model: nnx.Module,
        data: PyTree,
        train: bool = True,
        **kwargs,
    ) -> PyTree:
        """The model apply function that can be used in the loss function for
        simplification."""

        # state_vars = (getattr(self.state, svar) for svar in self.state_variables())
        # model = nnx.merge(state.graph_def, params, *(svar for svar in state_vars if svar is not None))
        if train:
            model.train()
            return model.augment(data)
        else:
            model.eval()
            return data

    def base_loss(self, logits, labels) -> tuple[jnp.ndarray, Metrics]:
        preds = logits.argmax(axis=-1)

        if self.config.loss_type == "cross_entropy":
            if jnp.issubdtype(labels.dtype, jnp.integer):
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                acc = (preds == labels).sum()
                acc2 = ((preds == labels) ** 2).sum()
                acc_count = (labels >= 0).sum()
            else:
                loss = optax.softmax_cross_entropy(logits, labels).mean()
                acc = (jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1).sum()
                acc2 = ((jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1) ** 2).sum()
                acc_count = labels.sum()
        elif self.config.loss_type == "binary_cross_entropy":
            labels = nn.one_hot(labels, self.config.num_classes) if labels.ndim == 1 else labels
            labels = labels.astype(jnp.float32)

            loss = optax.sigmoid_binary_cross_entropy(logits, labels > 0).sum()
            acc = (jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1).sum()
            acc2 = ((jnp.eye(labels.shape[-1])[preds] * labels).sum(axis=-1) ** 2).sum()
            acc_count = labels.sum()
        else:
            raise ValueError()

        metrics = {
            "acc": {
                "value": acc,
                "count": acc_count,
                "mode": LogMetricMode.MEAN,
                "log_mode": LogMode.ANY,
            },
            "acc_std": {
                "value": acc,
                "value2": acc2,
                "count": acc_count,
                "mode": LogMetricMode.STD,
                "log_mode": LogMode.EVAL,
            },
            "acc_max": {
                # a single item accuracy max is always 1, so this is assumed to be a batch max
                "value": acc / acc_count,
                "count": 1,
                "mode": LogMetricMode.MAX,
                "log_mode": LogMode.TRAIN,
                "log_freq": LogFreq.EPOCH,
            },
        }

        return loss, metrics

    def loss_function_linen(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict] | Dict]:
        """Loss function for image classification.

        Args:
            params: Parameters of the model.
            state: State of the trainer.
            batch: Batch of data. Assumes structure of SupervisedBatch or subclasses.
            rng: Key for random number generation.
            train: Whether the model is in training mode.

        Returns:
            Tuple of loss and tuple of mutable variables and metrics.
        """
        imgs = batch.input
        labels = batch.target

        # apply preprocessing (should have no variables)
        imgs, labels = self.preprocessing.apply(
            {}, (imgs, labels), method=self.preprocessing.preprocess
        )

        # apply augmentation (tied to model, but should have no own variables)
        if train:
            rng, aug_rng = jax.random.split(rng)
            imgs, labels = self.model.apply(
                {"params": params},
                (imgs, labels),
                rng={"augmentation": aug_rng},
                method=self.model.augment,
            )
        logits, mutable_variables = self.model_linen_apply(
            params=params, state=state, input=imgs, rng=rng, train=train, deterministic=not train
        )

        loss, metrics = self.base_loss(logits, labels)

        return loss, (mutable_variables, metrics)

    def loss_function(
        self,
        model: nnx.Module,
        batch: SupervisedBatch,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict]]:
        """Loss function for image classification.

        Args:
            params: Parameters of the model.
            state: State of the trainer.
            batch: Batch of data. Assumes structure of SupervisedBatch or subclasses.
            rng: Key for random number generation.
            train: Whether the model is in training mode.

        Returns:
            Tuple of loss and tuple of mutable variables and metrics.
        """
        imgs = batch.input
        labels = batch.target

        imgs, labels = self.preprocessing.preprocess((imgs, labels))
        imgs, labels = self.model_augment(model, data=(imgs, labels), train=train)

        logits = self.model_apply(
            model=model,
            inputs=imgs,
            train=train,
            deterministic=not train,
        )

        loss, metrics = self.base_loss(logits, labels)

        return loss, metrics
