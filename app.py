import gradio as gr
from gan_interface import create_latent_grid

import jax.numpy as jnp 
import flax 
import jax 
from typing import Any
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

generator_trained = restore_checkpoint(ckpt_dir= 'saved_models', target = None)

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


def interface(category, code_1, code_2):

    return create_latent_grid(
        num_images = 16,
        state = state,
        params = state.params,
        rng_key = jax.random.PRNGKey(0),
        image_dims=(4, 4),
        cat_idx = category,
        c1_value = code_1,
        c2_value = code_2,
    )

iface = gr.Interface(fn=interface,
    inputs=[gr.inputs.Dropdown(choices = [i for i in range(0,10)], type="value", default=None, label=None), gr.inputs.Slider(-2, 2), gr.inputs.Slider(-2, 2)],
    outputs= ["image"],
    live=True,
    title="Play with InfoGAN",
    description="Interactive InfoGAN model trained on MNIST. Code 1 controls the thickness of the generated samples while Code 2 controls the rotation.",
    )
iface.launch()