from models.generator import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from utils.create_latents_with_codes import create_latents_with_codes


def create_latent_grid(num_images, model, rng_key, image_dims=(10, 10)):

    # TODO: Ability to vary codes

    latent_var = create_latents_with_codes(
        num_noise=62, num_cts=2, num_cat=10, rng_key=rng_key, num_samples=num_images
    )

    output, _ = model.apply(
        {"params": params, "batch_stats": batch_stats},
        latent_var,
        mutable=["batch_stats"],
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

    plt.show()


if __name__ == "__main__":

    model = Generator()

    def initialized(key, latent_size, model):
        input_shape = (1, latent_size)

        @jax.jit
        def init(rng, shape):
            return model.init(rng, shape, train=True)

        variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
        return variables["params"], variables["batch_stats"]

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    params, batch_stats = initialized(rng, 74, model)

    create_latent_grid(100, model, rng_key=rng)
