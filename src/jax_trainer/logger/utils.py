from typing import Iterable

from jax import numpy as jnp

from jax_trainer.nnx_dummy import nnx


def flatten_dict(
    cfg: dict,
    separation_mark: str = ".",
):
    """Returns a nested OmegaConf dict as a flattened dict, merged with the
    separation mark.

    Example:
        With separation_mark == '.', {'data': {'this': 1, 'that': 2}} is returned as {'data.this': 1, 'data.that': 2}.

    Args:
        cfg (ConfigDict): The nested config dict.
        separation_mark (str, optional): The separation mark to use. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """
    cfgdict = dict(cfg)
    keys = list(cfgdict.keys())
    for key in keys:
        if isinstance(cfgdict[key], dict):
            flat_dict = flatten_dict(cfgdict.pop(key), separation_mark)
            for flat_key in flat_dict.keys():
                cfgdict[separation_mark.join([key, flat_key])] = flat_dict[flat_key]
    return cfgdict


def module_named_params(
    module: nnx.Module, recursive: bool = False
) -> Iterable[tuple[str, nnx.Param]]:
    if recursive:
        for submodule_name, submodule in module.iter_children():
            for named_param in module_named_params(submodule, recursive=recursive):
                yield (f"{submodule_name}.{named_param[0]}", named_param[1])

    for param_name in vars(module):
        potential_par = getattr(module, param_name)
        if isinstance(potential_par, nnx.Param) and potential_par.value is not None:
            yield param_name, potential_par


def count_parameters(module: nnx.Module) -> int:
    num_pars = 0
    for _, par in module_named_params(module=module, recursive=True):
        num_pars += par.size

    return num_pars


def count_parameters_linen(params: dict) -> int:
    num_pars = 0
    for key, par in params.items():
        if isinstance(par, jnp.ndarray):
            num_pars += par.size
        elif isinstance(par, dict):
            num_pars += count_parameters_linen(par)
    return num_pars
