# InfoGAN in Jax

## Description

InfoGAN (cite) proposes an updated loss function for GANs to learn disentangled (define) representations by adding an 'information regularization' term to the standard GAN loss function. The information regularization term approximates mutual information between a subset of the noise latents (called codes) and the output from the generator function. Including this term in the loss encourages these latent codes to connected to meaningful concepts such as scale, thickness, or rotation. 

## Running Code

Configuration is handled with [Hydra](https://hydra.cc/). The default MNIST config contains the following:

```yaml
training:
  epochs: 101
  workers: 4
  batch_size: 128
  weight_decay: 5e-4
  mixed_precision: False 
  discriminator_lr: 2e-4
  generator_lr: 1e-3
  lambda_cat: 1.0
  lambda_cts: 0.1

model:
  num_noise: 62
  num_cts_codes: 2
  num_cat_codes: 1
  num_categories: 10

data:
  dataset: 'MNIST'
```
To train InfoGAN on MNSIT, run:
```
python main.py
```
Training for 100 epochs with full precision takes around 35 minutes on an RTX 2080. 

## To do


2. Update docstrings
3. Host public WandB link (https://wandb.ai/bfattori/InfoGAN)

## Testing
```
python -m pytest
```

## References

- https://github.com/Natsu6767/InfoGAN-PyTorch is a very helpful repo.
- https://github.com/bilal2vec/jax-dcgan is a good reference for working with GANs in Jax. 
