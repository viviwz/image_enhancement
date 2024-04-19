"""dataloader for tif files

Author: Wei Zhao
Date: April 18, 2024
"""
import os
import numpy as np
import tifffile
from pathlib import Path
from itertools import chain
import glob
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch3dunet.datasets.hdf5 import _create_padded_indexes
import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, calculate_stats, mirror_pad, ConfigDataset
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TIFDataset')

def load_files_with_extensions(dataset_dir, extensions):
    return sorted([str(file) for ext in extensions for file in dataset_dir.glob(f"*{ext}")])

class TIFDataset(ConfigDataset):
    EXTENSIONS = ['.tif', '.tiff', '.TIF', '.TIFF']
    def __init__(self, raw_file_path, label_file_path, phase, slice_builder_config, transformer_config, weight_file_path=None, global_normalization=True):
        """
        Implementation of torch.utils.data.Dataset backed by the TIF files, which iterates over the raw and label datasets 
        patch by patch with a given stride. 

        Following the pytorch3dunet.datasets.AbstractHDF5Dataset style

        Args:
            raw_file_path (str): path to H5 file containing raw data as well as labels and per pixel weights
            label_file_path: groundtruth file 
            phase (str): 'train' for training, 'val' for validation, 'test' for testing
            slice_builder_config (dict): configuration of the SliceBuilder
            transformer_config (dict): data augmentation configuration
            weight_file_path (str or list, optional): path to file recording the per pixel weights
            global_normalization (bool, optional): if True, the mean and std of the raw data will be calculated over the whole dataset
    """
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.raw_file_path = raw_file_path
        self.label_file_path = label_file_path
        if weight_file_path:
            self.weight_internal_path = weight_file_path
        else:
            self.weight_internal_path = None

        self.halo_shape = slice_builder_config.get('halo_shape', [0, 0, 0])

        if global_normalization:
            logger.info('Calculating mean and std of the raw data...')
            raw = self.read_image(self.raw_file_path)
            stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

            if self.weight_internal_path  is not None:
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_transform = None

            self._check_volume_sizes()
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None

            # compare patch and stride configuration
            patch_shape = slice_builder_config.get('patch_shape')
            stride_shape = slice_builder_config.get('stride_shape')
            if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                logger.warning(f'Found non-zero halo shape {self.halo_shape}. '
                               f'In this case: patch shape and stride shape should be equal for optimal prediction '
                               f'performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!')

        # build slices
        raw = self.read_image(self.raw_file_path)
        label = self.read_image(self.label_file_path) if phase != 'test' else None
        weight_map = self.load_weights(self.weight_internal_path ) if self.weight_internal_path  is not None else None

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(raw, label, weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

        self._raw = None
        self._raw_padded = None
        self._label = None
        self._weight_map = None
    
    def read_image(self, file_path):
        """Read image array from file

        Args:
            file_path (str): path to image file
        """
        return tifffile.imread(file_path)
    
    def load_weights(self, weight_file_path):
        assert Path(weight_file_path).suffix()=='.npy', "Weight file must be '.npy'"

        return np.load(weight_file_path)
    
    def _check_volume_sizes(self):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        raw = self.read_image(self.raw_file_path)
        label = self.read_image(self.label_file_path)
        assert raw.ndim in [3,4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

        if self.weight_internal_path  is not None:
            weight_map = self.load_weights(self.weight_internal_path )
            assert weight_map.ndim in [3, 4], 'Weight map dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert _volume_shape(raw) == _volume_shape(weight_map), 'Raw and weight map have to be of the same size'
        
    def get_raw_patch(self, idx):
        if self._raw is None:
            self._raw = self.read_image(self.raw_file_path)
        return self._raw[idx]

    def get_label_patch(self, idx):
        if self._label is None:
            self._label = self.read_image(self.label_file_path)
        return self._label[idx]

    def get_weight_patch(self, idx):
        if self._weight_map is None:
            self._weight_map = self.load_weights(self.weight_internal_path)
        return self._weight_map[idx]

    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            self._raw_padded = mirror_pad(self.read_image(self.raw_file_path), self.halo_shape)
        return self._raw_padded[idx]
    
    def volume_shape(self):
        raw = self.read_image(self.raw_file_path)
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == 'test':
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[1:]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(raw_idx, self.halo_shape)
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            raw_patch_transformed = self.raw_transform(self.get_raw_padded_patch(raw_idx_padded))
            return raw_patch_transformed, raw_idx
        else:
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.get_label_patch(label_idx))
            if self.weight_internal_path is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.get_weight_patch(weight_idx))
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count            
    
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        # dataset_paths contain only directories, each directory is a path to a dataset, 
        # such as ["path/to/dataset1", "path/to/dataset2"]      
        dataset_dirs_list = phase_config['file_paths']

        # load (raw_image_file, gt_image_file, weight_file(optional)) pairs for all files in dataset
        # dataset directory contains ['raw_images', 'ground_truth', 'weights'(optional)]
        file_paths_list = []
        for dataset_dir in dataset_dirs_list:
            weights_dir = os.path.join(dataset_dir, "weights") if dataset_config.get('weight_map', False) else None
            file_paths_list += cls.load_dataset_files(dataset_dir, weights_dir=weights_dir)
        
        # create dataset for each file
        datasets = []
        for raw_file, gt_file, weight_file in file_paths_list:
            try:
                logger.info(f'Loading {phase} set from: {raw_file}...')
                dataset = cls(raw_file_path=raw_file,
                              label_file_path=gt_file,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              weight_file_path=weight_file,
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {raw_file}', exc_info=True)
        return datasets
    
    @classmethod
    def load_dataset_files(cls, dataset_dir, weights_dir=None):
        
        raw_images_dir = Path(dataset_dir) / "raw_images"
        ground_truth_dir = Path(dataset_dir) / "ground_truth" 
        # weight_dir = Path(dataset_dir) / "weights"

        find_existing_file = lambda files: next((file for file in files if os.path.isfile(file)), None)
        
        raw_image_files = load_files_with_extensions(raw_images_dir, cls.EXTENSIONS)
        pairs = []
        for raw_file in raw_image_files:
            gt_file = [os.path.join(ground_truth_dir, f"{Path(raw_file).stem}_gt{ext}") for ext in cls.EXTENSIONS]
            gt_file = find_existing_file(gt_file)
            
            weight_file = os.path.join(weight_dir, f"{Path(raw_file).stem}.npy") if weights_dir else None

            pairs.append((raw_file, gt_file, weight_file))
        return pairs

def create_train_loaders_from_config(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # create TIFDataset
    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = TIFDataset.create_datasets(loaders_config, phase='train')

    val_datasets = TIFDataset.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=num_workers)
    }

def create_test_loaders_from_config(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset
    test_datasets = TIFDataset.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         collate_fn=collate_fn)
