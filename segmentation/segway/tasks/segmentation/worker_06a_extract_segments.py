import os
import json
import logging
import sys
import time
import numpy as np

from funlib.segment.arrays import replace_values

import daisy
from daisy import Coordinate

from lut import LookupTable
from base_task import Database

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
# logging.getLogger('lut').setLevel(logging.DEBUG)


def segment_in_block(
        block,
        lut_dir,
        merge_function,
        thresholds,
        segmentations,
        fragments):

    logger.info("Received block %s" % block)

    logger.debug("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    logger.debug("%.3fs"%(time.time() - start))

    lut_db = LookupTable(filepath=lut_dir)

    for threshold in thresholds:

        segmentation = segmentations[threshold]

        logger.debug("Load local LUT...")
        start = time.time()
        start0 = start
        # local_lut = 'seg_frags2local_%s_%d/%d' % (merge_function, int(threshold*100), block.block_id)
        # local_lut = np.load(os.path.join(lut_dir, local_lut + ".npz"))['fragment_segment_lut']
        dataset = 'seg_frags2local_%s_%d' % (merge_function, int(threshold*100))
        local_lut = lut_db.load_lut(block, dataset=dataset)
        logger.debug("Found %d fragments" % len(local_lut[0]))
        logger.debug("%.3fs"%(time.time() - start))

        logger.debug("Relabelling fragments to local segments")
        start = time.time()
        relabelled = replace_values(fragments, local_lut[0], local_lut[1], inplace=False)
        logger.debug("%.3fs"%(time.time() - start))

        logger.debug("Load global LUT...")
        start = time.time()
        # global_lut = 'seg_local2global_%s_%d/%d' % (merge_function, int(threshold*100), block.block_id)
        # global_lut = np.load(os.path.join(lut_dir, global_lut + ".npz"))['fragment_segment_lut']
        dataset = 'seg_local2global_%s_%d' % (merge_function, int(threshold*100))
        global_lut = lut_db.load_lut(block, dataset=dataset)
        logger.debug("Found %d fragments" % len(global_lut[0]))
        logger.debug("%.3fs"%(time.time() - start))

        logger.debug("Relabelling fragments to global segments")
        start = time.time()
        relabelled = replace_values(relabelled, global_lut[0], global_lut[1], inplace=True)
        logger.debug("%.3fs"%(time.time() - start))

        logger.debug("Writing segments...")
        start = time.time()
        segmentation[block.write_roi] = relabelled
        logger.debug("%.3fs"%(time.time() - start))
        logger.info("Took %.3fs"%(time.time() - start0))


def extract_segmentation(
        fragments_file,
        fragments_dataset,
        lut_filename,
        lut_dir,
        merge_function,
        threshold,
        out_file,
        out_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
        run_type=None,
        **kwargs):

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)

    segment_in_block(
        roi_offset,
        roi_shape,
        lut_dir,
        merge_function,
        threshold,
        segmentation,
        fragments,
        lut)


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    if run_config.get('block_id_add_one_fix', False):
        daisy.block.Block.BLOCK_ID_ADD_ONE_FIX = True

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    daisy_client = daisy.Client()

    completion_db = Database(db_host, db_name, completion_db_name)

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    segmentations = {}
    for threshold in thresholds:
        ds = out_dataset + "_%.3f" % threshold
        segmentations[threshold] = daisy.open_ds(out_file, ds, mode='r+')

    while True:
        with daisy_client.acquire_block() as block:

            if block is None:
                break

            segment_in_block(
                block,
                lut_dir,
                merge_function,
                thresholds,
                segmentations,
                fragments)

            # recording block done in the database
            completion_db.add_finished(block.block_id)

            daisy_client.release_block(block)
