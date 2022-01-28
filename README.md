# InfoGAN in Jax

## Description

InfoGAN (cite) proposes an updated loss function for GANs to learn disentangled (define) representations by adding an 'information regularization' term to the standard GAN loss function. The information regularization term approximates mutual information between a subset of the noise latents (called codes) and the output from the generator function. Including this term in the loss ensures that the latent codes are connected to meaningful concepts such as scale, thickness, or rotation. 

## To do

1. Basic training loop done
    1.1 Loss functions - requires 3 different loss functions (DONE)
    1.2 Optimizers - AdamW probably - do this via ```weight_decay``` transform in optax - add param masking too

2. Code to generate samples and display images 
    2.1 Pretty much done I think

3. PyTorch dataloading
    3.1 Should be able to reuse most of the code from ResNets - Pretty much done

## Testing
```
python -m pytest
```

## References

https://github.com/Natsu6767/InfoGAN-PyTorch is a very helpful repo 