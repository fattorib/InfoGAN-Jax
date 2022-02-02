import gradio as gr
from gradio_utils.gan_interface import create_latent_grid

import jax.numpy as jnp
import flax
import jax
from typing import Any
from models.generator import Generator
from flax.training.checkpoints import restore_checkpoint
import matplotlib.pyplot as plt
from flax.training import train_state
import optax

if __name__ == '__main__':

    generator = Generator(dtype=jnp.float32)


    class TrainState(train_state.TrainState):
        batch_stats: Any = None
        weight_decay: Any = None
        dynamic_scale: flax.optim.DynamicScale = None


    generator_trained = restore_checkpoint(ckpt_dir="saved_models", target=None)

    state = TrainState.create(
        apply_fn=generator.apply,
        params=generator_trained["params"],
        tx=optax.adamw(learning_rate=0.1, weight_decay=0, b1=0.5),
        batch_stats=generator_trained["batch_stats"],
        weight_decay=generator_trained["weight_decay"],
        dynamic_scale=None,
    )


    def interface(category, code_1, code_2):

        return create_latent_grid(
            num_images=16,
            state=state,
            params=state.params,
            rng_key=jax.random.PRNGKey(0),
            image_dims=(4, 4),
            cat_idx=category if category != 'Random Selection' else None,
            c1_value=code_1,
            c2_value=code_2,
        )


    iface = gr.Interface(
        fn=interface,
        inputs=[
            gr.inputs.Dropdown(
                choices=['Random Selection'] + [i for i in range(0, 10)], type="value", default='Random Selection', label='Categorical Code - Digit Type'
            ),
            gr.inputs.Slider(-2, 2, default=0.0, label = 'Code 1 - Thickness'),
            gr.inputs.Slider(-2, 2, default=0.0, label = 'Code 2 - Rotation'),
        ],
        outputs=["image"],
        live=True,
        theme = 'grass',
        title="Play with InfoGAN",
        description="Interactive InfoGAN model trained on MNIST. Code 1 controls the thickness of the generated samples and Code 2 controls the rotation of the generated samples. The category of the generated digits can be changed with the categorical code variable. Note that the value here will not align with the actual digit created. This is as intended as InfoGAN is associating a one-hot encoded vector to a specific digit type. ",
    )
    iface.launch()
