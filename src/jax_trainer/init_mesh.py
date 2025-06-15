import jax
from jax.experimental import mesh_utils


def init_ddp_mesh() -> jax.sharding.Mesh:
    num_devices = jax.device_count()
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((num_devices,)), ("data",))
    return mesh
