import jax.numpy as jnp 
import numpy as np 
import flax 
import jax 
from typing import Any
from utils.create_generator_grid import create_latent_grid
from models.generator import Generator
from flax.training.checkpoints import restore_checkpoint
import matplotlib.pyplot as plt
from flax.training import train_state
import optax
generator = Generator(dtype=jnp.float32)

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None
    dynamic_scale: flax.optim.DynamicScale = None

generator_trained = restore_checkpoint(ckpt_dir= 'saved_checkpoints', target = None)

state = TrainState.create(
        apply_fn=generator.apply,
        params=generator_trained['params'],
        tx = optax.adamw(
        learning_rate=0.1, weight_decay=0, b1=0.5
        ),
        batch_stats=generator_trained['batch_stats'],
        weight_decay=generator_trained['weight_decay'],
        dynamic_scale= None,
    )



def create_latent_grid(
    num_images,
    state,
    params,
    rng_key,
    image_dims=(4, 4),
    cat_idx = None,
    c1_value = None,
    c2_value = None,
    ):

    num_noise=62
    num_cts=2
    num_cat=10
    # create irreducible noise
    z = jax.random.normal(rng_key, shape=(num_images, num_noise))

    if c1_value is not None:
        c = jnp.array(jnp.tile(c1_value, num_images)).reshape(
                num_images, 1
            )
        z = jnp.concatenate([z, c], axis=1)
    else:
        rng_key, subkey = jax.random.split(rng_key)
        c = jnp.tile(
            jax.random.uniform(key=subkey, shape=(1, 1), minval=-1, maxval=1),
            num_images,
        ).reshape(num_images, 1)
        z = jnp.concatenate([z, c], axis=1)

    if c2_value is not None:
        c = jnp.array(jnp.tile(c2_value, num_images)).reshape(
                num_images, 1
            )
        z = jnp.concatenate([z, c], axis=1)
    else:
        rng_key, subkey = jax.random.split(rng_key)
        c = jnp.tile(
            jax.random.uniform(key=subkey, shape=(1, 1), minval=-1, maxval=1),
            num_images,
        ).reshape(num_images, 1)
        z = jnp.concatenate([z, c], axis=1)

    # categorical 
    if cat_idx is not None:
        c = jnp.array(
        [
            jnp.tile(jax.nn.one_hot(cat_idx, num_classes=num_cat), num_images)
        ]
        ).reshape(num_images, num_cat)

        z = jnp.concatenate([z, c], axis=1)
    else:
        logit_probs = [1 / num_cat for _ in range(num_cat)]
        c_idx = jax.random.categorical(
            key=rng_key, logits=jnp.array(logit_probs), shape=(num_images,)
        )
        c = jax.nn.one_hot(c_idx, num_classes=num_cat)

        z = jnp.concatenate([z, c], axis=1)


    # This code should be fine 
    output = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        z,
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

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))


    return image_from_plot


# image = create_latent_grid(
#     num_images = 16,
#     state = state,
#     params = state.params,
#     rng_key = jax.random.PRNGKey(0),
#     image_dims=(4, 4),
#     cat_idx = 0,
#     c1_value = None,
#     c2_value = None,
# )

# plt.show()