import json
import logging
import sys
import os
import numpy as np

import daisy
from daisy import Coordinate

from lut import LookupTable
from base_task import Database

# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logging.getLogger('lut').setLevel(logging.DEBUG)


def replace_fragment_ids(
        db_host,
        db_name,
        fragments_file,
        lut_dir,
        merge_function,
        roi_offset,
        roi_shape,
        total_roi,
        thresholds,
        block,
        ):
    '''Compute local2local edges from local2frags edges and frags2local LUT (step 04a)'''

    logger.info(f"Received block {block}")

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)
    lut_db = LookupTable(filepath=lut_dir)

    for threshold in thresholds:

        '''Load local2frags edges of this block and replace the target frags
        with the super fragment node IDs'''

        # edges = 'edges_local2frags_%s_%d/%d.npz' % (merge_function, int(threshold*100), block_id)
        # edges = os.path.join(lut_dir, edges)
        # edges = np.load(edges)['edges']
        dataset = 'edges_local2frags_%s_%d' % (merge_function, int(threshold*100))
        edges = lut_db.load_edges(block, dataset=dataset)

        adj_blocks = generate_adjacent_blocks(roi_offset, roi_shape, total_roi)
        for adj_block in adj_blocks:
            # adj_block_id = block.block_id

            # lookup = 'seg_frags2local_%s_%d/%d.npz' % (merge_function, int(threshold*100), adj_block_id)
            # lut = os.path.join(
            #         lut_dir,
            #         lookup)
            # logging.debug("Reading %s.." % (lookup))
            # if not os.path.exists(lut):
            #     logging.info("Skipping %s.." % lookup)
            #     continue
            # lut = np.load(lut)['fragment_segment_lut']
            logger.debug(f'Reading {adj_block}')
            dataset = 'seg_frags2local_%s_%d' % (merge_function, int(threshold*100))
            try:
                lut = lut_db.load_lut(adj_block, dataset=dataset)
            except FileNotFoundError:
                logger.info(f'Error reading {adj_block} for {block}')
                continue

            frags2seg = {n: k for n, k in np.dstack((lut[0], lut[1]))[0]}

            for i in range(len(edges)):
                if edges[i][0] in frags2seg:
                    if edges[i][0] != frags2seg[edges[i][0]]:
                        edges[i][0] = frags2seg[edges[i][0]]
                if edges[i][1] in frags2seg:
                    if edges[i][1] != frags2seg[edges[i][1]]:
                        edges[i][1] = frags2seg[edges[i][1]]

        if len(edges):
            # np.unique doesn't work on empty arrays
            edges = np.unique(edges, axis=0)

        # final_edges = 'edges_local2local_%s_%d/%d.npz' % (merge_function, int(threshold*100), block_id)
        # out_file = os.path.join(lut_dir, final_edges)
        # logger.debug("Writing %s" % final_edges)
        # np.savez_compressed(out_file, edges=edges)
        dataset = 'edges_local2local_%s_%d' % (merge_function, int(threshold*100))
        lut = lut_db.save_edges(block, edges, dataset=dataset)


def generate_adjacent_blocks(roi_offset, roi_shape, total_roi):

    blocks = []
    current_block_roi = daisy.Roi(roi_offset, roi_shape)

    total_write_roi = total_roi.grow(-roi_shape, -roi_shape)

    for offset_mult in [
            Coordinate(-1, 0, 0),
            Coordinate(+1, 0, 0),
            Coordinate(0, -1, 0),
            Coordinate(0, +1, 0),
            Coordinate(0, 0, -1),
            Coordinate(0, 0, +1),
            ]:

        shifted_roi = current_block_roi.shift(roi_shape*offset_mult)
        if total_write_roi.intersects(shifted_roi):
            blocks.append(
                daisy.Block(total_roi, shifted_roi, shifted_roi))

    return blocks


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

    total_roi = daisy.Roi(total_roi_offset, total_roi_shape)

    while True:
        with daisy_client.acquire_block() as block:

            if block is None:
                break

            roi_offset = block.write_roi.get_offset()
            roi_shape = daisy.Coordinate(tuple(block_size))

            replace_fragment_ids(
                db_host=db_host,
                db_name=db_name,
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                merge_function=merge_function,
                block=block,
                total_roi=total_roi,
                roi_offset=roi_offset,
                roi_shape=roi_shape,
                thresholds=thresholds,
                )

            # recording block done in the database
            completion_db.add_finished(block.block_id)

            daisy_client.release_block(block)
