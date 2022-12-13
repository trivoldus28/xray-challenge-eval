import daisy
import sys
import logging
import numpy as np
import argparse

from funlib.segment.arrays import replace_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("fpath", type=str, help='')
    ap.add_argument("--include_zero", type=int, default=0)
    ap.add_argument("--exclude_labels", type=int, default=[], nargs='+')
    ap.add_argument("--input_ds_label", type=str, default="volumes/training_gt_labels")
    ap.add_argument("--input_ds_mask", type=str, default="volumes/training_gt_labels_mask")
    ap.add_argument("--output_ds_label", type=str, default="volumes/training_gt_unlabeled_mask")
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    out_file = fpath

    segment_ds = daisy.open_ds(out_file, input_ds_label)
    mask_ds = daisy.open_ds(out_file, input_ds_mask)

    out_ds = daisy.prepare_ds(
        out_file,
        output_ds_label,
        segment_ds.roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 3},
        delete=True
        )

    unlabeled_ndarray = np.ones(out_ds.shape, dtype=out_ds.dtype)

    replace_vals = exclude_labels
    if not include_zero:
        replace_vals.append(0)
    replace_vals = list(set(replace_vals))

    if len(replace_vals):
        labels_ndarray = segment_ds.to_ndarray()
        segment_by_foreground = replace_vals
        new_mask_values = [0]*len(replace_vals)
        replace_values(
            labels_ndarray,
            segment_by_foreground,
            new_mask_values,
            unlabeled_ndarray)

    unlabeled_ndarray = unlabeled_ndarray * mask_ds.to_ndarray()

    out_ds[out_ds.roi] = unlabeled_ndarray
