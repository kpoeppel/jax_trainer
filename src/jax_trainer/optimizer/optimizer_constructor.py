from typing import Any, Callable

import optax
from ml_collections import ConfigDict

from jax_trainer.utils import resolve_import_from_string


class OptimizerBuilder:
    """Class for building optimizers from config.

    Can be overwritten to add custom optimizers and learning rate schedules.
    """

    def __init__(self, optimizer_config: ConfigDict):
        self.optimizer_config = optimizer_config

    def build_optimizer(self, num_epochs: int = 0, num_train_steps_per_epoch: int = 0):
        """Build optimizer from config.

        Args:
            optimizer_config (ConfigDict): ConfigDict for optimizer.

        Returns:
            optax.GradientTransformation: Optimizer.
        """
        # Build elements of optimizer
        opt_class = self.build_optimizer_function()
        lr_schedule = self.build_lr_scheduler(
            num_epochs=num_epochs, num_train_steps_per_epoch=num_train_steps_per_epoch
        )
        pre_grad_trans, post_grad_trans = self.build_gradient_transformations()
        # Put everything together
        optimizer = optax.chain(*pre_grad_trans, opt_class(lr_schedule), *post_grad_trans)
        return optimizer, lr_schedule

    def build_optimizer_function(self):
        """Build optimizer class function from config.

        By default, it supports Adam, AdamW, and SGD. To add custom optimizers, overwrite the
        function build_extra_optimizer_function.

        Returns:
            Callable: Optimizer class function.
        """
        # Build optimizer class
        optimizer_name = self.optimizer_config.name
        optimizer_name = optimizer_name.lower()
        optimizer_params = self.optimizer_config.get("params", ConfigDict())
        opt_class = None
        if optimizer_name == "adam":
            opt_class = lambda sched: optax.adam(
                sched,
                b1=optimizer_params.get("beta1", 0.9),
                b2=optimizer_params.get("beta2", 0.999),
                eps=optimizer_params.get("eps", 1e-8),
            )
        elif optimizer_name == "adamw":
            opt_class = lambda sched: optax.adamw(
                sched,
                b1=optimizer_params.get("beta1", 0.9),
                b2=optimizer_params.get("beta2", 0.999),
                eps=optimizer_params.get("eps", 1e-8),
                weight_decay=self.optimizer_config.get("transforms", ConfigDict()).get(
                    "weight_decay", 0.0
                ),
            )
        elif optimizer_name == "sgd":
            opt_class = lambda sched: optax.sgd(
                sched,
                momentum=optimizer_params.get("momentum", 0.0),
                nesterov=optimizer_params.get("nesterov", False),
            )
        else:
            opt_class = self.build_extra_optimizer_function(
                optimizer_name=optimizer_name, optimizer_params=optimizer_params
            )
        return opt_class

    def build_extra_optimizer_function(self, optimizer_name: str, optimizer_params: ConfigDict):
        """Function that can be overwritten by subclasses to add custom optimizers. By default, it
        raises a ValueError.

        Args:
            optimizer_name (str): Name of the optimizer.
            optimizer_params (ConfigDict): ConfigDict for optimizer parameters.
        """
        try:
            optimizer_class = resolve_import_from_string(f"optax.{optimizer_name}")
            return lambda sched: optimizer_class(sched, **optimizer_params)
        except ImportError:
            raise ValueError(f"Unknown optimizer {optimizer_name}.")

    def build_lr_scheduler(self, num_epochs: int = 0, num_train_steps_per_epoch: int = 0):
        """Build learning rate schedule from config.

        By default, it supports constant, cosine decay, exponential decay, and warmup cosine decay.
        To add custom learning rate schedules, overwrite the function build_extra_lr_scheduler.

        Args:
            num_epochs (int, optional): Number of epochs. Defaults to 0.
            num_train_steps_per_epoch (int, optional): Number of training steps per epoch. Defaults to 0.

        Returns:
            Callable: Learning rate schedule function.
        """
        # Build learning rate schedule
        lr = float(self.optimizer_config.lr)
        scheduler_config = self.optimizer_config.get("scheduler", ConfigDict())
        scheduler_name = scheduler_config.get("name", None)
        decay_steps = scheduler_config.get("decay_steps", num_epochs * num_train_steps_per_epoch)
        lr_schedule = None
        if scheduler_name is None or scheduler_name == "constant":
            lr_schedule = optax.constant_schedule(lr)
        elif scheduler_name == "cosine_decay":
            assert decay_steps > 0, "decay_steps must be positive"
            lr_schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=decay_steps,
                alpha=scheduler_config.get("alpha", 0.0),
            )
        elif scheduler_name == "exponential_decay":
            # Exponential decay with cooldown and warmup
            cooldown = scheduler_config.get("cooldown_steps", 0)
            warmup = scheduler_config.get("warmup_steps", 0)
            num_schedule_steps = decay_steps - cooldown - warmup
            if scheduler_config.get("decay_rate", None) is not None:
                decay_rate = scheduler_config.decay_rate
            else:
                assert decay_steps > 0, "decay_steps must be positive"
                if scheduler_config.get("end_lr", None) is not None:
                    end_lr_factor = scheduler_config.end_lr / lr
                elif scheduler_config.get("end_lr_factor", None) is not None:
                    end_lr_factor = scheduler_config.end_lr_factor
                else:
                    raise ValueError("Either end_lr or end_lr_factor must be specified.")
                decay_rate = end_lr_factor ** (1.0 / num_schedule_steps)
            lr_schedule = optax.warmup_exponential_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                decay_rate=decay_rate,
                warmup_steps=warmup,
                transition_steps=scheduler_config.get("transition_steps", 1),
                staircase=scheduler_config.get("staircase", False),
            )
            if cooldown > 0:
                assert decay_steps > 0, "decay_steps must be positive"
                end_lr = lr * (decay_rate**num_schedule_steps)
                lr_schedule = optax.join_schedules(
                    schedules=[
                        lr_schedule,
                        optax.linear_schedule(
                            init_value=end_lr,
                            end_value=0.0,
                            transition_steps=cooldown,
                        ),
                    ],
                    boundaries=[decay_steps - cooldown],
                )
        elif scheduler_name == "warmup_cosine_decay":
            assert decay_steps > 0, "decay_steps must be positive"
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                decay_steps=decay_steps,
                warmup_steps=scheduler_config.warmup_steps,
                end_value=scheduler_config.get("end_value", 0.0),
            )
        else:
            lr_schedule = self.build_extra_lr_scheduler(
                scheduler_name=scheduler_name,
                scheduler_config=scheduler_config,
                num_epochs=num_epochs,
                num_train_steps_per_epoch=num_train_steps_per_epoch,
            )
        return lr_schedule

    def build_extra_lr_scheduler(
        self,
        scheduler_name: str,
        scheduler_config: ConfigDict,
        num_epochs: int = 0,
        num_train_steps_per_epoch: int = 0,
    ):
        """Function that can be overwritten by subclasses to add custom learning rate schedules. By
        default, it raises a ValueError.

        Args:
            scheduler_name (str): Name of the learning rate schedule.
            scheduler_config (ConfigDict): ConfigDict for the learning rate schedule.
            num_epochs (int, optional): Number of epochs. Defaults to 0.
            num_train_steps_per_epoch (int, optional): Number of training steps per epoch. Defaults to 0.
        """
        try:
            lr_schedule_class = resolve_import_from_string(f"optax.{scheduler_name}")
            return lr_schedule_class(**scheduler_config)
        except ImportError:
            raise ValueError(f"Unknown learning rate schedule {scheduler_name}.")

    def build_gradient_transformations(self):
        """Build gradient transformations from config.

        By default, it supports gradient clipping by norm and value, and weight decay. To add custom
        gradient transformations, overwrite this function and call super().build_gradient_transformations()
        in it. We distinguish between pre- and post-optimizer gradient transformations. Pre-optimizer
        gradient transformations are applied before the optimizer, e.g. gradient clipping. Post-optimizer
        gradient transformations are applied after the optimizer.

        Returns:
            Tuple[List[Callable], List[Callable]]: Tuple of pre-optimizer and post-optimizer gradient transformations.
        """
        # Gradient transformation
        optimizer_name = self.optimizer_config.name
        optimizer_name = optimizer_name.lower()
        transform_config = self.optimizer_config.get("transforms", ConfigDict())
        grad_trans = {"pre": [], "post": []}

        def add_grad_trans(config: Any, gt_fn: Callable):
            if isinstance(config, (float, int, str, bool)):
                gt = gt_fn(config)
                grad_trans["pre"].append(gt)
            elif isinstance(config, ConfigDict):
                gt = gt_fn(config.value)
                if config.get("before_optimizer", True):
                    grad_trans["pre"].append(gt)
                else:
                    grad_trans["post"].append(gt)

        if transform_config.get("grad_clip_norm", None) is not None:
            add_grad_trans(transform_config.grad_clip_norm, optax.clip_by_global_norm)
        if transform_config.get("grad_clip_value", None) is not None:
            add_grad_trans(transform_config.grad_clip_value, optax.clip)
        if transform_config.get("weight_decay", 0.0) > 0.0 and optimizer_name not in ["adamw"]:
            add_grad_trans(transform_config.weight_decay, optax.add_decayed_weights)

        return grad_trans["pre"], grad_trans["post"]
