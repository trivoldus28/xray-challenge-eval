from __future__ import print_function
import sys

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('gunpowder.nodes.generic_train').setLevel(logging.CRITICAL)  # disable logging

import os
from gunpowder import *
from reject import Reject
import os
import math
import json
import numpy as np
import logging
from mknet import create_network
from gunpowder.jax import Train as JaxTrain
from tqdm import tqdm

global cache_size
global snapshot_every
cache_size = 40
snapshot_every = 5000

logging.basicConfig(level=logging.INFO)

import jax
n_devices = jax.local_device_count()
# n_devices = 1

dense_samples = [
    '../../../data/xpress-challenge.zarr'
]

raw_ds = 'volumes/training_raw'
labels_ds = 'volumes/training_gt_labels'
labels_mask_ds = 'volumes/training_gt_labels_mask'
unlabeled_mask_ds = 'volumes/training_gt_unlabeled_mask'

raw = ArrayKey('RAW')
labels = ArrayKey('GT_LABELS')
labels_mask = ArrayKey('GT_LABELS_MASK')
unlabeled_mask = ArrayKey('GT_UNLABELED_MASK')
affs = ArrayKey('PREDICTED_AFFS')
gt_affs = ArrayKey('GT_AFFINITIES')
gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
gt_affs_scale = ArrayKey('GT_AFFINITIES_SCALE')
affs_gradient = ArrayKey('AFFS_GRADIENT')

neighborhood = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-3, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-9, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-27, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])


def train_until(max_iteration, num_workers, model, batch_size):

    print(f'cache_size={cache_size}')

    trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    voxel_size = Coordinate((33, 33, 33))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabeled_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[labels],
        affs_gradient: request[labels],
        #gt_affs: request[labels],
    })

    padding_voxels = 32
    source_padding = Coordinate([padding_voxels*voxel for voxel in voxel_size])

    def create_source(sample):

        src = ZarrSource(
                sample,
                datasets={
                    raw: raw_ds,
                    labels: labels_ds,
                    labels_mask: labels_mask_ds,
                    unlabeled_mask: unlabeled_mask_ds,
                },
                array_specs={
                    raw: ArraySpec(interpolatable=True),
                    labels: ArraySpec(interpolatable=False),
                    labels_mask: ArraySpec(interpolatable=False),
                    unlabeled_mask: ArraySpec(interpolatable=False),
                }
            )

        src += Pad(raw, None)
        src += Pad(labels, source_padding)
        src += Pad(labels_mask, source_padding)
        src += Pad(unlabeled_mask, source_padding)

        src += RandomLocation()

        src += Reject(
            mask=unlabeled_mask,
            min_masked=0.1,
        )

        src += Normalize(raw)

        return src

    data_sources = tuple(
        create_source(sample) for sample in dense_samples
    )

    train_pipeline = (
        data_sources +
        # RandomProvider(probabilities=total_sample_weights) +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[10, 10, 10],  # worked for 30nm
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi/2.0],
            subsample=2,
            ) +

        SimpleAugment() +

        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        NoiseAugment(raw, var=0.01) +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabeled_mask,
            affinities_mask=gt_affs_mask) +

        # doesn't work for fractional GT affs
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask,
            ) +

        IntensityScaleShift(raw, 2, -1) +
        # add raw "channel"
        Unsqueeze([raw]) +
        # add batch
        Stack(batch_size) +
        # Unsqueeze([raw, gt_affs, gt_affs_scale]) +
        PreCache(
            cache_size=cache_size,
            num_workers=num_workers-1) +
        JaxTrain(
            model=model,
            inputs={
                'raw': raw,
                'gt': gt_affs,
                'mask': gt_affs_scale,
            },
            outputs={
                'affs': affs,
                'grad': affs_gradient
            },
            log_dir='log',
            save_every=20000,
            #keep_n_checkpoints=20,
            ) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                unlabeled_mask: 'volumes/labels/unlabeled',
                affs_gradient: 'volumes/affs_gradient'
            },
            every=snapshot_every,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=100)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        # for i in range(max_iteration - trained_until):
        for i in tqdm(range(max_iteration - trained_until)):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":

    model = create_network()

    if 'debug' in sys.argv:
        # iteration = 5
        # num_workers = 6
        # cache_size = 5
        # snapshot_every = 1
        iteration = 10
        num_workers = 2
        cache_size = 1
        snapshot_every = 1
    elif 'debug_perf' in sys.argv:
        iteration = 1000
        num_workers = 24
        # num_workers = 36
        snapshot_every = 10
    else:
        try:
            iteration = int(sys.argv[1])
            num_workers = int(sys.argv[2])
        except:
            iteration = 1000000
            num_workers = 16*n_devices
            # cache_size = 24*n_devices
            # num_workers = 24

        # cache_size = 40*batch_size
        cache_size = num_workers*2
        # cache_size = 1
        # cache_size = 80

    batch_size = 1*n_devices
    train_until(iteration, num_workers, model, batch_size)
