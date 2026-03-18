import os, csv
import argparse

import pandas as pd

import torch
import torch.nn as nn
import torchvision

from logger import Logger, CSVBatchLogger
from train import train

from temp import CUBDataset, DRODataset


n_epochs = 300
weight_decay = 0.0001

def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')

def prepare_data():
    full_dataset = CUBDataset(
        root_dir='cub',
        target_name='waterbird_complete95',
        confounder_names=['forest2water2'],
        model_type='resnet50',
        augment_data=False
    )
    splits = ['train', 'val', 'test']
    subsets = full_dataset.get_splits(splits, train_frac=1.0)
    return [DRODataset(
        subsets[split],
        process_item_fn=None,
        n_groups=full_dataset.n_groups,
        n_classes=full_dataset.n_classes,
        group_str_fn=full_dataset.group_str
    ) for split in splits]

def main():
    logger = Logger('./logs/log.txt', 'w')

    train_data, val_data, test_data = prepare_data()

    loader_kwargs = {'batch_size': 128, 'num_workers': 4, 'pin_memory': True}

    train_loader = train_data.get_loader(train=True, reweight_groups=False, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    model = torchvision.models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
    print(f"ResNet50 chargé — fc: {d} -> {n_classes} classes")

    logger.flush()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    train_csv_logger = CSVBatchLogger('./logs/train.csv', train_data.n_groups, mode='w')
    val_csv_logger = CSVBatchLogger('./logs/val.csv', train_data.n_groups, mode='w')
    test_csv_logger = CSVBatchLogger('./logs/test.csv', train_data.n_groups, mode='w')
    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, weight_decay, n_epochs)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

if __name__ == '__main__':
    main()