#!/usr/bin/env python3
""" Inference for E2E-Spot """

import os
import argparse
import re
import torch

from dataset.frame_process import ActionSeqVideoDataset
from util.io import load_json
from util.dataset import load_classes
from train_f3set_f3ed import F3Set, evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model dir')
    parser.add_argument('frame_dir', help='Path to the frame dir')
    parser.add_argument('-s', '--split',
                        choices=['train', 'val', 'test', 'challenge'],
                        required=True)

    parser.add_argument('-d', '--dataset',
                        help='Dataset name if not inferrable from the config')
    return parser.parse_args()


def get_best_epoch(model_dir, key='val_edit'):
    data = load_json(os.path.join(model_dir, 'loss.json'))
    best = max(data, key=lambda x: x[key])
    return best['epoch']


def get_last_epoch(model_dir):
    regex = re.compile(r'checkpoint_(\d+)\.pt')

    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    assert last_epoch >= 0
    return last_epoch


def main(model_dir, frame_dir, split, dataset):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as fp:
        print(fp.read())

    config = load_json(config_path)
    if os.path.isfile(os.path.join(model_dir, 'loss.json')):
        best_epoch = get_best_epoch(model_dir)
        print('Best epoch:', best_epoch)
    else:
        best_epoch = get_last_epoch(model_dir)

    if dataset is None:
        dataset = config['dataset']
    else:
        if dataset != config['dataset']:
            print('Dataset mismatch: {} != {}'.format(
                dataset, config['dataset']))

    classes = load_classes(os.path.join('data', dataset, 'elements.txt'))

    model = F3Set(len(classes), config['feature_arch'], config['temporal_arch'], clip_len=config['clip_len'],
                  step=config['stride'], window=config['window'], use_ctx=config['use_ctx'],
                  multi_gpu=config['gpu_parallel'])

    model.load(torch.load(os.path.join(
        model_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

    split_path = os.path.join('data', dataset, '{}.json'.format(split))
    split_data = ActionSeqVideoDataset(classes, split_path, frame_dir, config['clip_len'], 
                                       overlap_len=config['clip_len'] // 2, crop_dim=config['crop_dim'], 
                                       stride=config['stride'])

    evaluate(model, split_data, classes)


if __name__ == '__main__':
    main(**vars(get_args()))

