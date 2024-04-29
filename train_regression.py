import sys
import random
import torch
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.config import copy_config, load_config
from pytorch3dunet.unet3d.config import _load_config_yaml as load_config_yaml
from trainer import create_trainer


logger = get_logger('TrainingSetup')

def train_3dunet_regression(config, config_path=None):
    # Load and log experiment configuration
    if isinstance(config, str):
        config_path = config
        config = load_config_yaml(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config)

    # Copy config file
    if config_path:
        copy_config(config, config_path)

    # Start training
    trainer.fit()

def main():
    # Load and log experiment configuration
    if len(sys.argv)>1:
        config_path = sys.argv[1]
    else:
        config, config_path = load_config()
    logger.info(config_path)

    train_3dunet_regression(config_path)


if __name__ == '__main__':
    main()