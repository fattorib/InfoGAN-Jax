from models.generator import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from utils.create_latents_with_codes import create_latents_with_codes, create_latents_manual_categorical, create_latents_manual_cts


def create_latent_grid(num_images, state, params, rng_key, image_dims=(10, 10), categorical_idx = None, cts_idx = None):

    if categorical_idx is None and cts_idx is None:
        latent_var = create_latents_with_codes(
            num_noise=62, num_cts=2, num_cat=10, rng_key=rng_key, num_samples=num_images
        )

    elif categorical_idx is not None:
        assert cts_idx is None
        latent_var = create_latents_manual_categorical(num_noise=62, num_cts=2, num_cat=10, rng_key = rng_key, num_samples = 100)

    elif cts_idx is not None:
        assert categorical_idx is None
        latent_var = create_latents_manual_cts(num_noise=62, num_cts=2, num_cat=10, rng_key = rng_key, cts_idx = cts_idx, num_samples = 100)

    output = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        latent_var,
        mutable=False,
        train=False,
    )

    np_images = jax.device_get(output).reshape(num_images, 28, 28)

    ncols = image_dims[0]
    nrows = image_dims[1]

    assert ncols * nrows == num_images, "Check your plot dimensions"

    fig = plt.figure()
    axes = [
        fig.add_subplot(nrows, ncols, r * ncols + c + 1)
        for r in range(0, nrows)
        for c in range(0, ncols)
    ]

    i = 0
    for ax in axes:
        ax.imshow(np_images[i, :], cmap="gray")
        i += 1

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig



