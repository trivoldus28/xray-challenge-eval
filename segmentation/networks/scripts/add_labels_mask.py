import daisy
import sys
import logging
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("fpath", type=str, help='')
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    if fpath.endswith(".zarr") or fpath.endswith(".n5"):
        out_file = fpath

    segment_ds = daisy.open_ds(out_file, 'volumes/training_gt_labels')

    out = daisy.prepare_ds(
        out_file,
        "volumes/training_gt_labels_mask",
        segment_ds.roi,
        segment_ds.voxel_size,
        np.uint8,
        compressor={'id': 'zlib', 'level': 3},
        delete=True
        )

    arr_shape = segment_ds.roi.get_shape() / segment_ds.voxel_size

        # arr = np.ones(arr_shape, dtype=out.dtype)
        # if unfinished_sections is not None:
        #     for s in unfinished_sections:
        #         arr[s, :, :] = 0

    out[out.roi] = 1
