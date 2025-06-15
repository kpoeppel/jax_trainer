# JAX-Trainer: Lightning-like API for JAX with Flax

This repository is an extension of Phillip Lippe's `jax_trainer`. The goal is to provide a Lightning-like API for JAX with Flax. The API is inspired by [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and has as basic element a `TrainerModule`. This module implements common training and evaluation loops, and can be used to train a model with a few lines of code. This fork extends the library for use of both `flax.linen` and `flax.nnx` based models. The train loop can be extended via callbacks, which are similar to Lightning's callbacks.

While in original repo, `ml-collections` was used for configuration, this was refactored to use the [compoconf](https://github.com/kpoeppel/compoconf.git) library. The benefits are type-safety and easy compositionality via interface / protocol patterns.

## Installation

In future, the package will be available on PyPI. For now, you can install it from source:

```bash
git clone https://github.com/kpoeppel/jax_trainer.git
cd jax_trainer
pip install -e .
```

## Usage

In the following, we will go through the main API choices in the library. In most cases, the user will only need to implement a loss function in a subclass of `TrainerModule` for each task, besides the actual models in Flax. The training loop can be further customized via callbacks. All modules are then configured via a YAML file and can be trained with a few lines of code.

### TrainerModule API

The `jax_trainer.trainer.TrainerModule` has been written with the goal to be as flexible as possible while still providing a simple API for training and evaluation.

The main aspects of the trainer is to:

- **Initialize the model**: The model is initialized via the `init_model` function. This function is called at the beginning of the training and evaluation. The function can be overwritten by the user to implement custom initialization logic.
- **Handling the TrainState**: The trainer keeps a `TrainState` which contains the model state, the optimizer state, the random number generator state, and any mutable variables. The `TrainState` is updated after each training step and can be used to resume training from a checkpoint.
- **Logging**: The trainer provides a simple logging interface by allowing the train and evaluation functions to return dictionaries of metrics to log.
- **Saving and loading checkpoints**: The trainer provides functions to save and load checkpoints. The checkpoints are saved as `TrainState`s and can be used to resume training or to evaluate a model. A pre-trained model can also be loaded by simply calling `TrainerModule.load_from_checkpoint`, similar to the API in Lightning.
- **Training and evaluation**: The trainer provides functions to train and evaluate a model. The training and evaluation loops can be extended via callbacks, which are called at different points during the training and evaluation.

As a user, the main function that needs to be implemented for each individual task is `loss_function_linen(...)` in the case of `linen` and `loss_function(...)` in the case of `nnx`. This function takes as input the model parameters and state, the batch of data, a random number generator key, and a boolean indicating whether its training or not. The function needs to return the loss, as well as a tuple of mutable variables and optional metrics. For the `nnx` case the state is part of the `nnx.Module`, therefore it only that the `model`, `batch`, and `train` switch and returns main `loss` and `metrics`. The `TrainerModule` then takes care of the rest, which includes wrapping it into a training and evaluation function, performing gradient transformations, and calling it in a loop. The choice of `nnx` model vs `linen` model is in the `model_mode` `TrainerConfig` switch. Additionally, to provide a unified interface with other functions like initialization, the subclass needs to implement `batch_to_input` which, given a batch, returns the input to the model. Model and Optimizer have to implement the `BaseModelNNX`/`BaseModelLinen` and `OptimizerInterface`, and via `compoconf`, they can automatically register their configuration type and can be instantiated from that config.
The following example shows a simple trainer module for image classification:

```python
class ImgClassifierTrainer(TrainerModule):

    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.input

    def loss_function_linen(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict]]:
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_linen_apply(
            params=params, state=state, input=imgs, rng=rng, train=train
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()
        metrics = {"acc": acc}
        return loss, (mutable_variables, metrics)

    def loss_function(
        self,
        model: nnx.Module,
        batch: SupervisedBatch,
        train: bool = True,
    ) -> Tuple[Any, Dict]:
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_apply(
            params=params, state=state, input=imgs, rng=rng, train=train
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()
        metrics = {"acc": acc}
        return loss, metrics
```

### Logging API

The `metrics` dictionary returned by the loss function is used for logging. By default, the logger supports to log values every *N* training steps and/or per epoch. For more options on the logger, see the configuration documentation below.

Further, the logging of each metric can be customized by providing additional options in the `metrics` dictionary. For each metric, the following options are available:

- `mode`: The mode of the metric describes how it should be aggregated over the epoch or batches. The different options are summarized in the `jax_trainer.logger.LogMetricMode` enum. Currently, the following modes are available:
  - `LogMetricMode.MEAN`: The mean of the metric is logged.
  - `LogMetricMode.SUM`: The sum of the metric is logged.
  - `LogMetricMode.SINGLE`: A single value of the metric is used, namely the last one logged.
  - `LogMetricMode.MAX`: The max of the metric is logged.
  - `LogMetricMode.MIN`: The min of the metric is logged.
  - `LogMetricMode.STD`: The standard deviation of the mtric is logged.
  - `LogMetricMode.CONCAT`: The values of the metric are concatenated. Note that in this case, the metric is not logged to the tool of choice (e.g. Tensorboard or WandB), but is only provided in the full metric dictionary, which can be used as input to callbacks.
- `log_freq`: The frequency of logging the metric. The options are summarized in `jax_trainer.logger.LogFreq` and are the following:
  - `LogFreq.EPOCH`: The metric is logged only once per epoch.
  - `LogFreq.STEP`: The metric is logged only per *N* steps.
  - `LogFreq.ANY`: The metric is logged both per epoch and per *N* steps.
- `log_mode`: The training mode in which the metric should be logged. This allows for different metrics to be logged during training, validation and/or testing. The options are summarized in the enum `jax_trainer.logger.LogMode` with the options:
  - `LogMode.TRAIN`: The metric is logged during training.
  - `LogMode.VAL`: The metric is logged during validation.
  - `LogMode.TEST`: The metric is logged during testing.
  - `LogMode.EVAL`: The metric is logged during both validation and testing.
  - `LogMode.ANY`: The metric is logged during any of the above modes.

### Callback API

The `TrainerModule` provides a callback API which is similar to the one in Lightning. The callbacks are called at different points during the training and evaluation. Each callback can implement the following methods:

- `on_training_start`: Called at the beginning of the training.
- `on_training_end`: Called at the end of the training.
- `on_filtered_training_epoch_start`: Called at the beginning of each training epoch.
- `on_filtered_training_epoch_end`: Called at the end of each training epoch.
- `on_filtered_validation_epoch_start`: Called at the beginning of the validation.
- `on_filtered_validation_epoch_end`: Called at the end of the validation.
- `on_test_epoch_start`: Called at the beginning of the testing.
- `on_test_epoch_end`: Called at the end of the testing.

The training and validation functions with `filtered` in the name are only called every *N* epochs, where *N* is the value of `every_n_epochs` in the callback configuration.

The following callbacks are pre-defined:

- `ModelCheckpoint`: Saves the model and optimizer state after validation. This checkpoint can be used to resume training or to evaluate the model. It is implemented using `orbax` and is similar to the `ModelCheckpoint` in Lightning.
- `LearningRateMonitor`: Logs the learning rate at the beginning of each epoch. This is similar to the `LearningRateMonitor` in Lightning.

For configuring the callbacks, also for custom callbacks, see the configuration documentation below.

### Dataset API

The dataset API abstracts the data loading with PyTorch, using numpy arrays for storage. Each dataset needs to provide a train, validation and test loader. As return type, we use `flax.struct.dataclass`es, which are similar to PyTorch's `NamedTuple`s. These dataclasses can be used in jit-compiled functions and are therefore a good fit for JAX. Additionally, each batch should define a `size` attribute, which is used for taking the correct average across batches in evaluation. For an example, see the `jax_trainer.datasets.examples` module.

## Configuration

The configuration is done via a YAML file. It consists of four main sections: `trainer`, `model`, `optimizer`, and `dataset`. The `trainer` section configures the `TrainerModule` and the callbacks. The `model` section configures the model, which is implemented by the user. The `optimizer` section configures the optimizer and the learning rate scheduler. The `dataset` section configures the dataset. The following example shows a configuration for training a simple MLP on CIFAR10:

```yaml
seed: 42
num_gpus: 1
trainer:
  class_name: ImgClassifierTrainer
  train_epochs: 5
  check_val_every_n_epoch: 1
  debug: False
  enable_progress_bar: True
  seed: 42
  log_grad_norm: True
  detect_nans: True
  num_classes: 10
  logger:
    class_name: Logger
    log_dir: tests/checkpoints/BuildTrainerTest/
    log_steps_every: 50
    tool_config:
      class_name: TensorboardToolLogger
      name: ""
      save_dir: tests/checkpoints/BuildTrainerTest/
      use_timestamp_version: false
      version: "0"
    log_file_verbosity: warning
  callbacks:
    ModelCheckpoint:
      class_name: ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 1
      save_optimizer_state: False
    LearningRateMonitor:
      class_name: LearningRateMonitor
      every_n_epochs: 1
    JAXProfiler:
      class_name: JAXProfiler
      every_n_minutes: 60
      first_step: 10
      profile_n_steps: 20
    # GradientSpikeMonitor:
    #   every_n_epochs: 1
    #   log_to_disk: True
    #   ema_decay: 0.99
    #   threshold: 3.0
model:
  class_name: SimpleClassifier
  c_hid: 32
  num_classes: 10
  act_fn: gelu
  batch_norm: True
optimizer:
  class_name: AdamW
  b1: 0.9
  b2: 0.999
  eps: 1.0e-08
  learning_rate:
    class_name: WarmupCosineDecaySchedule
    decay_factor: 0.1
    decay_steps: 1555
    init_value: 0.0
    peak_value: 0.001
    steps: 1755
    warmup_steps: 200
  nesterov: false
  weight_decay:
    class_name: WeightDecay
    mode: whitelist
    parameter_regex_exclude: ""
    parameter_regex_include: ((.*weight$)|(.*kernel$))
    value: 0.01
dataset:
  class_name: CIFAR10Dataset
  data_dir: data/
  global_batch_size: 128
  local_batch_size: 128
  num_workers: 0
```

In the following, we will go through the different sections and explain the configuration options.

### Trainer

The `trainer` section configures the `TrainerModule` and the callbacks. The `TrainerModule` is configured via the following options:

- `class_name`: Name of the `TrainerModule` class. Currently, the following classes are available:
  - `ImgClassifierTrainer`: Trainer for image classification tasks.
  - `TrainerModule`: Base class for implementing custom trainers.
    For own-implemented trainers, the name of the class is the class_name and the class must be registered as implementing the interface TrainerModule via `compoconf`
- `train_epochs`: Number of training epochs.
- `check_val_every_n_epoch` (optional): Number of epochs between validation checks (default: 1). If set to `0`, no validation is performed.
- `debug` (optional): If `True`, the trainer is run in debug mode (default: False). This means that the training and validation steps are not jitted and can be easier analysed in case of an error.
- `enable_progress_bar` (optional): If True, a progress bar is shown during training and validation (default: True).
- `seed`: Seed for the initialization, model state, etc.
- `log_grad_norm` (optional): If True, the gradient norm is logged during training.
- `detect_nans` (optional): If True, the trainer will detect NaNs in the loss and gradients.
- `num_classes` (optional): Number of classes in the dataset. This is only required for some trainers, such as `ImgClassifierTrainer`.
- `logger`: Configuration of the logger. This is optional and in case of not being provided, a default logger is created. The following options are available:
  - `log_dir` (optional): Directory where the logging files are stored. If not provided, a default directory based on the model name and version is created.
  - `log_steps_every` (optional): Number of training steps between logging (default: 50). If set to `0`, logging is only performed per epoch. Otherwise, both per-epoch and per-step logging is performed.
  - `tool_config` (optional): Configuration of the logging tool. The following options are available:
    - `class_name`: Name of the logging tool class. Currently, the following tools are available:
      - `TensorboardToolLogger`: Logging to TensorBoard.
      - `WandBToolLogger`: Logging to Weights & Biases.
    - `name` (optional): Name of the experiment.
    - `save_dir` (optional): Directory where the logging files are stored.
    - `use_timestamp_version` (optional): If True, a timestamp is added to the version number.
    - `version` (optional): Version number of the experiment.
  - `log_file_verbosity` (optional): Verbosity of the logging file. Possible values are `debug`, `info`, `warning`, and `error`. By default, the verbosity is set to `info`.

The `callbacks` section configures the callbacks. The key of a callback is its name (if its a default one in `jax_trainer`) or arbitrary description. In case of the latter, the attribute `class_name` needs to be added, with the respective class path, e.g. `class_name: MyCallback`. Each callback has its own config and parameters. The following callbacks are pre-defined:

- `ModelCheckpoint`: Saves the model and optimizer state after validation.
  - `monitor` (optional): Metric to monitor (default: `val/loss`).
  - `mode` (optional): Mode of the metric (default: `min`). Possible values are `min`, `max`, and `auto`.
  - `save_top_k` (optional): Number of best models to save (default: `1`).
  - `save_optimizer_state` (optional): If True, the optimizer state is saved as well (default: `False`).
- `LearningRateMonitor`: Logs the learning rate at the beginning of each epoch.
  - `every_n_epochs` (optional): Number of training epochs between logging (default: `1`).
- `ConfusionMatrixCallback`: Logs the confusion matrix of a classifier after validation and testing. Requires the metric key `conf_matrix` to be logged.
  - `normalize` (optional): If True, the confusion matrix is normalized (default: `True`).
  - `cmap` (optional): Colormap of the confusion matrix (default: `Blues`).
  - `figsize` (optional): Size of the figure (default: `(8, 8)`).
  - `format` (optional): Format of the in-place text in each cell (default: `'.2%'` when normalized, else `'d'`).
  - `dpi` (optional): Dots per inch of the figure (default: `100`).
  - `every_n_epochs` (optional): Number of training epochs between logging (default: `1`).

### Model

The `model` section configures the model. It is a object of the config class corresponding to the model class that should be used.
The model class should implement BaseModelNNX or BaseModelLinen interfaces and register its config via `compoconf`'s `register`.

### Optimizer

The `optimizer` section configures the optimizer and the learning rate scheduler. The section has three main sub-sections:

#### Optimizer class

- Config class of the optimizer. Currently, the following optimizers are available:
  - `AdamW`
  - `SGD`
  - `Lamb`

Each optimizer has its own specific parameters that can be configured.

#### Learning rate scheduler

The learning rate scheduler is optional and is by default constant. The `learning_rate` field can be a float for a constant learning rate, or a configuration for a scheduler. The following schedulers are available:

- `ConstantSchedule`
- `CosineSchedule`
- `LinearSchedule`
- `ExponentialSchedule`
- `ConcatSchedule`
- `WarmupCosineDecaySchedule`

Each scheduler has its own specific parameters that can be configured.

#### Gradient Transformations

The gradient transformations are optional and can be used to transform the gradients before applying them to the model parameters. They are defined under the key `transforms`. The following transformations are available:

- `WeightDecay`
- `GradClipNorm`
- `GradClipValue`

Each transformation has its own specific parameters that can be configured, including a `before_optimizer` flag to specify if the transformation is applied before or after the optimizer step.

### Dataset

The `dataset` section configures the dataset and the data loading.

- `class_name`: Name of the dataset class. Currently, the following datasets are available:
  - `CIFAR10Dataset`
  - `MNISTDataset`

Each dataset has its own specific parameters that can be configured, such as:

- `data_dir` (optional): Directory where the dataset is stored (default: `data/`).
- `global_batch_size` (optional): Global batch size to use during training, validation and testing (default: `128`).
- `local_batch_size` (optional): Local batch size to use during training, validation and testing (default: `128`).
- `num_workers` (optional): Number of workers to use for data loading (default: `4`).
- `normalize` (optional): If True, the images are normalized (default: `True`).
- `val_size` (optional): Number of samples to use for the validation set (default: `5000`).
- `split_seed` (optional): Seed for splitting the dataset into train and validation sets (default: `42`).
- `pin_memory` (optional): If True, the data loaders will copy Tensors into device/CUDA pinned memory before returning them (default: `True`).
- `prefetch_factor` (optional): Number of batches loaded in advance by each worker (default: `4`).

## Contributing

Contributions are welcome! Before contributing code, please install the pre-commit hooks with:

```bash
pip install pre-commit
pre-commit install
```

This will run the linter and formatter on every commit.

If you have any questions, feel free to open an issue or contact me directly.
