import os
import time
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm
import torch
import torch.nn as nn
import tifffile
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.predictor import _is_2d_model
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.utils import SliceBuilder, remove_padding
from pytorch3dunet.unet3d.config import copy_config, load_config
from pytorch3dunet.unet3d.config import _load_config_yaml as load_config_yaml

from predictor import RegressionPredictor
from dataloader import create_test_loaders_from_config


logger = get_logger("Predict Regression")


def predict(config):
    # Load configuration
    if isinstance(config, str):
        config = load_config_yaml(config)

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available

    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    # Create predictor
    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    # remove the predictor config
    out_channels = config['model'].get('out_channels')
    predictor = RegressionPredictor(model, output_dir, out_channels)

    for test_loader in create_test_loaders_from_config(config):
        predictor(test_loader)

def main():
    config, config_path = load_config()
    logger.info(config)
    predict(config_path)

if __name__ == "__main__":
    main()

