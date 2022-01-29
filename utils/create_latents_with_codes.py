import jax
import jax.numpy as jnp
import operator


def create_latents_with_codes(num_noise, num_cts, num_cat, rng_key, num_samples):
    """Create a latent variable to feed into the generator

    Args:
        num_noise (int): number of 'noise' variables to use. For InfoGAN MNIST example, defaults to 62
        num_cts (int): number of continuous 'codes' to use. For InfoGAN MNIST example, defaults to 2
        num_cat (int): number of category 'codes' to use. For InfoGAN MNIST example, defaults to 10 categories
        rnk_key (*): jax prng key
        num_samples (int): number of latent samples to generate

    """

    # Noise
    z = jax.random.normal(rng_key, shape=(num_samples, num_noise))

    # Continuous code
    for _ in range(num_cts):
        # Need to split rng keys for each continuous code
        rng_key, subkey = jax.random.split(rng_key)

        c = jax.random.uniform(key=subkey, shape=(num_samples, 1), minval=-1, maxval=1)
        z = jnp.concatenate([z, c], axis=1)

    # Categorical code
    logit_probs = [1 / num_cat for _ in range(num_cat)]
    c_idx = jax.random.categorical(
        key=rng_key, logits=jnp.array(logit_probs), shape=(num_samples,)
    )

    c = jax.nn.one_hot(c_idx, num_classes=num_cat)

    z = jnp.concatenate([z, c], axis=1)

    return z


def create_latents_manual_categorical(
    num_noise, num_cts, num_cat, rng_key, num_samples=100
):
    """Create a latent variable to feed into the generator.

    Args:

    """

    # Noise
    z = jax.random.normal(rng_key, shape=(num_samples, num_noise))

    # Continuous code
    for _ in range(num_cts):
        rng_key, subkey = jax.random.split(rng_key)

        # Per paper, keep cts codes fixed
        c = jnp.tile(
            jax.random.uniform(key=subkey, shape=(1, 1), minval=-1, maxval=1),
            num_samples,
        ).reshape(num_samples, 1)
        z = jnp.concatenate([z, c], axis=1)

    c = jnp.array(
        [
            jnp.tile(jax.nn.one_hot(i, num_classes=num_cat), num_cat)
            for i in range(0, num_cat)
        ]
    ).reshape(num_samples, num_cat)

    z = jnp.concatenate([z, c], axis=1)

    return z


def create_latents_manual_cts(
    num_noise, num_cts, num_cat, rng_key, cts_idx, num_samples=100
):
    """Create a latent variable to feed into the generator.

    Args:

    """

    # Noise
    z = jax.random.normal(rng_key, shape=(num_samples, num_noise))

    # Continuous code
    for idx in range(num_cts):
        if idx == cts_idx:
            c = jnp.array(jnp.tile(jnp.linspace(-2, 2, 10), num_cat)).reshape(
                num_samples, 1
            )
            z = jnp.concatenate([z, c], axis=1)

        else:
            rng_key, subkey = jax.random.split(rng_key)
            c = jnp.tile(
                jax.random.uniform(key=subkey, shape=(1, 1), minval=-1, maxval=1),
                num_samples,
            ).reshape(num_samples, 1)
            z = jnp.concatenate([z, c], axis=1)
    c = jnp.array(
        [
            jnp.tile(jax.nn.one_hot(i, num_classes=num_cat), num_cat)
            for i in range(0, num_cat)
        ]
    ).reshape(num_samples, num_cat)

    z = jnp.concatenate([z, c], axis=1)

    return z
