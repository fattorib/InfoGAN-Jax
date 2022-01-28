from email import generator
import flax.linen as nn
import numpy
import optax
import functools

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

from models.discriminator import Discriminator
from models.generator import Generator
from models.recognition import Recognition

# Loss
from loss import binary_cross_entropy_loss, cross_entropy_loss, q_cts_loss

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
        root=f"{get_original_cwd()}/data/MNIST",
        train=True,
        download=False,
        transform=transform_train,
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.batch_size,
        pin_memory=False,
    )

    model_dtype = (
        jnp.float32 if cfg.training.mixed_precision.dtype == "False" else jnp.float16
    )

    # Setup WandB logging here
    wandb_run = wandb.init(project="InfoGAN")
    wandb.config.epochs = cfg.training.epochs
    wandb.config.batch_size = cfg.training.batch_size
    wandb.config.weight_decay = cfg.training.weight_decay

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Split rng init key for each of our models
    init_rng_disc, init_rng_gen, init_rng_q = jax.random.split(init_rng, num=3)

    noise_size = (
        cfg.model.num_noise
        + cfg.model.num_cts_codes
        + cfg.model.num_cat_codes * cfg.model.num_categories
    )

    generator = Generator()

    discriminator = Discriminator(filter_list=[64, 128, 1024])

    q_network = Recognition(
        filter_list=[64, 128, 1024, 128],
        num_cts_codes=cfg.model.num_cts_codes,
        num_cat=cfg.model.num_categories,
    )

    generator_state = create_train_state(
        rng=init_rng_gen,
        var_size=noise_size,
        learning_rate=cfg.training.generator_lr,
        weight_decay=cfg.training.weight_decay,
        model=generator,
    )

    q_state = create_train_state(
        rng=init_rng_q,
        var_size=28,
        learning_rate=cfg.training.generator_lr,
        weight_decay=cfg.training.weight_decay,
        model=q_network,
    )

    discriminator_state = create_train_state(
        rng=init_rng_disc,
        var_size=28,
        learning_rate=cfg.training.discriminator_lr,
        weight_decay=cfg.training.weight_decay,
        model=discriminator,
    )


def initialized_discriminator(key, image_size, model):
    input_shape = (1, image_size, image_size, 1)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]


def initialized_generator(key, variable_size, model):
    input_shape = (1, variable_size)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]


def create_train_state(rng, init_func, var_size, learning_rate, weight_decay, model):
    """Creates initial `TrainState`."""
    params, batch_stats = init_func(rng, var_size, model)

    # Mask for BN, bias params
    mask = jax.tree_map(lambda x: x.ndim != 1, params)
    tx = (
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay, mask=mask),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        weight_decay=weight_decay,
        dynamic_scale=flax.optim.DynamicScale() if model.dtype == jnp.float16 else None,
    )
    return state


@jax.jit
def train_discriminator_step(state, real_batch, generated_batch):
    """Train the discriminator for a single step"""

    def loss_fn(params):
        scores_real, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            real_batch,
            mutable=["batch_stats"],
            train=True,
        )

        scores_fake, new_state = new_state.apply_fn(
            {"params": params, "batch_stats": new_state.batch_stats},
            generated_batch,
            mutable=["batch_stats"],
            train=True,
        )

        loss_real = binary_cross_entropy_loss(
            scores=scores_real, labels=jnp.ones(cfg.training.batch_size)
        )

        loss_fake = binary_cross_entropy_loss(
            scores=scores_fake, labels=jnp.zeros(cfg.training.batch_size)
        )

        return loss_real + loss_fake, (scores_real, scores_fake, new_state)

    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)

    else:

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)

    scores_real, scores_fake, new_state = aux[1]

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_state["batch_stats"],
    )

    disc_loss = compute_discriminator_metrics(
        scores_real=scores_real, scores_fake=scores_fake
    )

    metrics = {"Discriminator Loss": disc_loss}

    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        state = state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin), state.opt_state, state.opt_state
            ),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin), state.params, state.params
            ),
        )

    return state, metrics


def compute_discriminator_metrics(
    *,
    scores_real,
    scores_fake,
):
    loss_real = binary_cross_entropy_loss(
        scores=scores_real, labels=jnp.ones(cfg.training.batch_size)
    )

    loss_fake = binary_cross_entropy_loss(
        scores=scores_fake, labels=jnp.zeros(cfg.training.batch_size)
    )

    return loss_real + loss_fake
