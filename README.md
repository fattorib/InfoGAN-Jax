# InfoGAN in Jax

## To do

1. Basic training loop done
    1.1 Loss functions - requires 3 different loss functions
    1.2 Optimizers - AdamW probably - do this via ```weight_decay``` transform in optax

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