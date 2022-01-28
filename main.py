import flax.linen as nn
import optax
# Flax imports
from typing import Any
import flax
from flax.training import train_state
import jax.numpy as jnp
import jax

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Hydra
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Utils
from utils.flax_utils import (
    NumpyLoader,
    FlattenAndCast,
)

# Loss
from loss import (
    binary_cross_entropy_loss, 
    cross_entropy_loss, 
    q_cts_loss
)

import wandb

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None
    dynamic_scale: flax.optim.DynamicScale = None



@hydra.main(config_path="conf", config_name="config_imagenette")
def main(config: DictConfig):

    global cfg

    cfg = config

    # ---------- Data Loading ---------- #

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            FlattenAndCast(),
        ]
    )

    train_dataset = MNIST(
        root=f"{get_original_cwd()}/data/MNIST", train=True, download=False, transform=transform_train
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.batch_size,
        pin_memory=False,
    )

    
    model_dtype = jnp.float32 if cfg.training.mixed_precision.dtype == "False" else jnp.float16

    # Setup WandB logging here
    wandb_run = wandb.init(project="Flax Torch")
    wandb.config.epochs = cfg.training.epochs
    wandb.config.batch_size = cfg.training.batch_size
    wandb.config.weight_decay = cfg.training.weight_decay

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)



# TODO: needs updating
def initialized_disc(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]
    
# TODO: needs updating
def initialized_gen(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]



# TODO: needs updating
def create_train_state(rng, momentum, learning_rate_fn, weight_decay, model):
    """Creates initial `TrainState`."""
    params, batch_stats = initialized(rng, 32, model)

    #TODO: Add masking for BN params
    #Technically this is AdamW - look at AdamW docstring to see how this is done
    tx = optax.sgd(learning_rate=learning_rate_fn, momentum=momentum, nesterov=True)


    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        weight_decay=weight_decay,
        dynamic_scale=flax.optim.DynamicScale() if model.dtype == jnp.float16 else None,
    )
    return state