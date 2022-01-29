import numpy as np
import flax.linen as nn
import numpy
import optax
import functools

from jax import random

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
from utils.create_latents_with_codes import create_latents_with_codes
from utils.create_generator_grid import create_latent_grid

from models.discriminator import Discriminator
from models.generator import Generator
from models.recognition import Q_head


# Loss
from loss import binary_cross_entropy_loss, cross_entropy_loss, q_cts_loss

import wandb


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None
    dynamic_scale: flax.optim.DynamicScale = None


@hydra.main(config_path="conf", config_name="config_MNIST")
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
        download=True,
        transform=transform_train,
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers,
        pin_memory=False,
    )

    model_dtype = (
        jnp.float32 if cfg.training.mixed_precision == "False" else jnp.float16
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

    q_network = Q_head(
        filter_size=128,
        num_cts_codes=cfg.model.num_cts_codes,
        num_cat=cfg.model.num_categories,
    )

    state_g = create_train_state(
        rng=init_rng_gen,
        init_func=initialize_generator,
        var_size=noise_size,
        learning_rate=cfg.training.generator_lr,
        weight_decay=cfg.training.weight_decay,
        model=generator,
    )

    state_q = create_train_state(
        rng=init_rng_q,
        init_func=initialize_Q_head,
        var_size=1024,
        learning_rate=cfg.training.generator_lr,
        weight_decay=cfg.training.weight_decay,
        model=q_network,
    )

    state_d = create_train_state(
        rng=init_rng_disc,
        init_func=initialize_discriminator,
        var_size=28,
        learning_rate=cfg.training.discriminator_lr,
        weight_decay=cfg.training.weight_decay,
        model=discriminator,
    )

    del init_rng_disc, init_rng_q, init_rng_gen

    for epoch in range(0, cfg.training.epochs):
        state_d, state_g, state_q, epoch_metrics_np = train_epoch(
            state_d, state_g, state_q, rng, train_loader
        )

        print(
            f"train epoch: {epoch}, discriminator loss: {epoch_metrics_np['Discriminator Loss']:.4f}, generator loss: {epoch_metrics_np['Generator Loss']:.4f}"
        )

        if epoch % 10 == 0:

            # Create basic samples
            image_generated = create_latent_grid(
                100, state_g, state_g.params, rng_key=rng
            )

            image1 = wandb.Image(image_generated, caption="Generator Samples")

            image_generated = create_latent_grid(
                100, state_g, state_g.params, rng_key=rng, categorical_idx=0
            )
            image2 = wandb.Image(
                image_generated, caption="Generator Samples (varying categorical code)"
            )

            image_generated = create_latent_grid(
                100, state_g, state_g.params, rng_key=rng, cts_idx=0
            )
            image3 = wandb.Image(
                image_generated, caption="Generator Samples (varying cts c_1)"
            )

            image_generated = create_latent_grid(
                100, state_g, state_g.params, rng_key=rng, cts_idx=1
            )
            image4 = wandb.Image(
                image_generated, caption="Generator Samples (varying cts c_2)"
            )

            wandb.log(
                {
                    "discriminator loss": epoch_metrics_np["Discriminator Loss"],
                    "generator loss": epoch_metrics_np["Generator Loss"],
                    "loss": (
                        epoch_metrics_np["Discriminator Loss"]
                        + epoch_metrics_np["Generator Loss"]
                    ),
                    "generator samples": image1,
                    "generator samples (varying categorical code)": image2,
                    "generator samples (varying cts c_1)": image3,
                    "generator samples (varying cts c_2)": image4,
                }
            )

        else:
            wandb.log(
                {
                    "discriminator loss": epoch_metrics_np["Discriminator Loss"],
                    "generator loss": epoch_metrics_np["Generator Loss"],
                    "loss": (
                        epoch_metrics_np["Discriminator Loss"]
                        + epoch_metrics_np["Generator Loss"]
                    ),
                }
            )


def initialize_discriminator(key, image_size, model):
    input_shape = (1, image_size, image_size, 1)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True, with_head=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]


def initialize_Q_head(key, filter_size, model):
    input_shape = (1, 1, 1, filter_size)

    @jax.jit
    def init(rng, shape):
        return model.init(
            rng,
            shape,
            train=True,
        )

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]


def initialize_generator(key, variable_size, model):
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
    tx = optax.adamw(
        learning_rate=learning_rate, weight_decay=weight_decay, mask=mask, b1=0.5
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
def loss_disc(params_d, params_g, state_g, state_d, real_batch, rng):

    z = create_latents_with_codes(
        cfg.model.num_noise,
        cfg.model.num_cts_codes,
        cfg.model.num_categories,
        rng,
        num_samples=real_batch.shape[0],
    )

    generated_batch, g_new_state = state_g.apply_fn(
        {"params": params_g, "batch_stats": state_g.batch_stats},
        z,
        mutable=["batch_stats"],
        train=True,
    )

    scores_real, d_new_state = state_d.apply_fn(
        {"params": params_d, "batch_stats": state_d.batch_stats},
        real_batch,
        mutable=["batch_stats"],
        train=True,
        with_head=True,
    )

    scores_fake, d_new_state = state_d.apply_fn(
        {"params": params_d, "batch_stats": state_d.batch_stats},
        generated_batch,
        mutable=["batch_stats"],
        train=True,
        with_head=True,
    )

    loss_real = binary_cross_entropy_loss(
        logit=scores_real, label=jnp.ones(real_batch.shape[0])
    )

    loss_fake = binary_cross_entropy_loss(
        logit=scores_fake, label=jnp.zeros(real_batch.shape[0])
    )

    return loss_real + loss_fake, (d_new_state, g_new_state)


@jax.jit
def loss_generator(params_g, params_q, params_d, state_g, state_q, state_d, rng):
    z = create_latents_with_codes(
        cfg.model.num_noise,
        cfg.model.num_cts_codes,
        cfg.model.num_categories,
        rng,
        num_samples=cfg.training.batch_size,
    )

    c_cts = z[:, cfg.model.num_noise : cfg.model.num_noise + cfg.model.num_cts_codes]

    c_cat = z[:, cfg.model.num_noise + cfg.model.num_cts_codes :]

    generated_batch, g_new_state = state_g.apply_fn(
        {"params": params_g, "batch_stats": state_g.batch_stats},
        z,
        mutable=["batch_stats"],
        train=True,
    )

    scores, d_new_state = state_d.apply_fn(
        {"params": params_d, "batch_stats": state_d.batch_stats},
        generated_batch,
        mutable=["batch_stats"],
        train=True,
        with_head=True,
    )

    loss = binary_cross_entropy_loss(
        logit=scores, label=jnp.ones(cfg.training.batch_size)
    )

    # This might be a dumb way to do things - turning off batch stats so we don't update this twice
    features_out = state_d.apply_fn(
        {"params": params_d, "batch_stats": state_d.batch_stats},
        generated_batch,
        mutable=False,
        train=False,
        with_head=False,
    )

    q_out, q_new_state = state_q.apply_fn(
        {"params": params_q, "batch_stats": state_q.batch_stats},
        features_out,
        mutable=["batch_stats"],
        train=True,
    )

    q_logits, q_mu, q_var, = (
        q_out[0],
        q_out[1],
        q_out[2],
    )

    q_loss_categorical = cross_entropy_loss(q_logits=q_logits, q_codes=c_cat)

    q_loss_cts = q_cts_loss(
        q_mu=q_mu, q_var=q_var, y=c_cts.reshape(-1, cfg.model.num_cts_codes)
    )

    return loss + q_loss_cts + q_loss_categorical, (
        g_new_state,
        q_new_state,
        d_new_state,
    )


@jax.jit
def train_step(state_d, state_g, state_q, batch, rng):

    # 1. Compute discriminator loss

    dynamic_scale = state_d.dynamic_scale
    if dynamic_scale:
        # grad_fn = dynamic_scale.value_and_grad(loss_disc, has_aux=True)
        # dynamic_scale, is_fin, aux, grads = grad_fn(
        #     state_d.params, state_g.params, state_g, state_d, batch, rng
        # )
        raise NotImplementedError

    else:
        grad_fn = jax.value_and_grad(loss_disc, has_aux=True)
        (discriminator_loss, (state_d_new, state_g_new)), grads = grad_fn(
            state_d.params, state_g.params, state_g, state_d, batch, rng
        )

    state_d = state_d.apply_gradients(
        grads=grads,
        batch_stats=state_d_new["batch_stats"],
    )

    if dynamic_scale:
        raise NotImplementedError
        # # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # # params should be restored (= skip this step).
        # state_d = state_d.replace(
        #     opt_state=jax.tree_multimap(
        #         functools.partial(jnp.where, is_fin),
        #         state_d.opt_state,
        #         state_d.opt_state,
        #     ),
        #     params=jax.tree_multimap(
        #         functools.partial(jnp.where, is_fin), state_d.params, state_d.params
        #     ),
        # )

    # 2. Compute

    dynamic_scale = state_q.dynamic_scale
    if dynamic_scale:
        raise NotImplementedError

    else:
        grad_fn = jax.value_and_grad(loss_generator, argnums=(0, 1), has_aux=True)
        (generator_loss, (state_g_new, state_q_new, state_d_new)), grads = grad_fn(
            state_g.params,
            state_q.params,
            state_d.params,
            state_g,
            state_q,
            state_d,
            rng,
        )

        grads_g, grads_q = grads

    state_q = state_q.apply_gradients(
        grads=grads_q, batch_stats=state_q_new["batch_stats"]
    )

    state_g = state_g.apply_gradients(
        grads=grads_g, batch_stats=state_g_new["batch_stats"]
    )

    if dynamic_scale:
        raise NotImplementedError

    metrics = {
        "Discriminator Loss": discriminator_loss,
        "Generator Loss": generator_loss,
    }

    return state_d, state_g, state_q, metrics


def train_epoch(state_d, state_g, state_q, rng, dataloader):
    """Train for a single epoch."""
    batch_metrics = []

    for batch, _ in dataloader:
        new_rng, subrng = random.split(rng)
        state_d, state_g, state_q, metrics = train_step(
            state_d,
            state_g,
            state_q,
            batch,
            subrng,
        )
        batch_metrics.append(metrics)
        rng = new_rng

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state_d, state_g, state_q, epoch_metrics_np


if __name__ == "__main__":
    main()
