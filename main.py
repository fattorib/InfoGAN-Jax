import flax.linen as nn
import optax
from flax.training import train_state

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Hydra
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Utils
from utils.flax_utils import (
    NumpyLoader,
    FlattenAndCast,
)


# ---------- Data Loading ---------- #

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        FlattenAndCast(),
    ]
)

train_dataset = MNIST(
    root="data/MNIST", train=True, download=False, transform=transform_train
)


train_loader = NumpyLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=False,
)
