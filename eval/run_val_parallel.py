from concurrent import futures

import daisy

from eval import run_eval
from database import Database


def eval_one(network, checkpoint, threshold, skeleton_file, roi, db_name):
    print(f'running: {network} {checkpoint} {threshold}')
    db = Database(db_name)
    segmentation_file = f'../segmentation/outputs/validation/setup03/{checkpoint}/output.zarr'
    segmentation_ds = f'volumes/segmentation_0.{int(threshold*1000):03}'
    res = run_eval(skeleton_file, segmentation_file, segmentation_ds, roi, downsampling=3)
    res.pop('gt_graph')
    res.pop('split_edges')
    res.pop('merged_edges')
    db.add_score(network=network, checkpoint=checkpoint, threshold=threshold, scores_dict=res)
    print(f'done: {network} {checkpoint} {threshold}')

if __name__ == '__main__':

    db_name = 'validation_results'
    db = Database(db_name)

    networks = ['setup03']
    # checkpoints = [990000]
    # checkpoints = [1000000, 420000, 440000, 460000, 480000, 490000, 960000, 970000, 980000]
    # checkpoints = [200000, 220000, 300000, 320000, 400000, 500000, 580000, 600000, 680000]
    # checkpoints += [700000, 780000, 800000, 880000, 900000, 1640000, 1660000]
    checkpoints = [240000, 260000, 340000, 360000, 520000, 540000, 620000, 640000, 720000, 740000, 820000, 840000, 1600000, 1620000, 1680000, 1700000]
    checkpoints += [280000, 380000, 560000, 660000, 760000, 860000]

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    skeleton_file = '../data/skeletons/XPRESS_validation_skels.npz'
    roi_begin = '8316,8316,8316'
    roi_shape = '23067,23067,23067'
    roi_begin = [float(k) for k in roi_begin.split(',')]
    roi_shape = [float(k) for k in roi_shape.split(',')]
    roi = daisy.Roi(roi_begin, roi_shape)

    ress = []
    with futures.ProcessPoolExecutor(max_workers=32) as tpe:
        for network in networks:
            for checkpoint in checkpoints:
                for threshold in thresholds:
                    ress.append(tpe.submit(eval_one, network, checkpoint, threshold, skeleton_file, roi, db_name))
    ress = [r.result() for r in ress]

    print(f'done: {network} {checkpoint} {threshold}')

