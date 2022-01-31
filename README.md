# InfoGAN in Jax

## Description

InfoGAN (cite) proposes an updated loss function for GANs to learn disentangled (define) representations by adding an 'information regularization' term to the standard GAN loss function. The information regularization term approximates mutual information between a subset of the noise latents (called codes) and the output from the generator function. Including this term in the loss encourages these latent codes to connected to meaningful concepts such as scale, thickness, or rotation. 

## To do

1. Get mixed precision working 
    - Not sure how to update multiple states. Might be easy with argnums to dynamic_scale.value_and_grad
2. Update docstrings
3. Host public WandB link (https://wandb.ai/bfattori/InfoGAN)
4. Tests for losses

## Testing
```
python -m pytest
```

## References

https://github.com/Natsu6767/InfoGAN-PyTorch is a very helpful repo 
https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb
