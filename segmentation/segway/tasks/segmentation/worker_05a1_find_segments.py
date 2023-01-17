import daisy
import os
import json
import logging
from funlib.segment.arrays import replace_values
import sys
import time
import numpy as np

import daisy
from daisy import Coordinate

from lut import LookupTable
from util import enumerate_blocks_in_chunks
from base_task import Database

# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logging.getLogger('lut').setLevel(logging.DEBUG)


def remap_in_block(
        block,
        block_size,
        total_roi,
        lut_dir,
        merge_function,
        threshold,
        global_lut=None,
        chunk_size=None):
    '''Remap local subsegment IDs to global segment IDs using the global
    connected component LUTs computed in step 04c.
    This step is only necessary for computing whole-dataset segmentation
    (as opposed to a block-wise segmentation).'''

    logging.info("Received block %s" % block)

    if global_lut is None:
        global_lut = load_global_lut(threshold, lut_dir, merge_function)

    lut_db = LookupTable(filepath=lut_dir)

    blocks = enumerate_blocks_in_chunks(
        block, block_size, chunk_size, total_roi)

    local_nodes_list = []
    for local_block in blocks:

        # nodes_file = 'nodes_%s_%d/%d.npz' % (
        #     merge_function, int(threshold*100), local_block.block_id)
        # nodes_file = os.path.join(lut_dir, nodes_file)
        # logging.info("Loading nodes %s" % nodes_file)
        # local_nodes_list.append(np.load(nodes_file)['nodes'])

        dataset = 'nodes_%s_%d' % (merge_function, int(threshold*100))
        local_nodes = lut_db.load_nodes(local_block, dataset=dataset)
        local_nodes_list.append(local_nodes)

    lens = [len(l) for l in local_nodes_list]
    chunk_start_index = [sum(lens[0:i+1]) for i in range(len(lens))]
    chunk_start_index.insert(0, 0)

    local_nodes_chunks = np.concatenate(local_nodes_list)

    # logging.info("Remapping nodes %s" % nodes_file)
    start = time.time()
    remapped_chunks = replace_values(local_nodes_chunks, global_lut[0], global_lut[1])
    # print("%.3fs" % (time.time() - start))

    for i, sub_block in enumerate(blocks):

        local_nodes = local_nodes_chunks[chunk_start_index[i]:chunk_start_index[i+1]]
        remapped = remapped_chunks[chunk_start_index[i]:chunk_start_index[i+1]]

        # remove self-referencing entries
        non_self_refs = local_nodes != remapped
        local_nodes = local_nodes[non_self_refs]
        remapped = remapped[non_self_refs]

        lut = np.array([local_nodes, remapped])

        # write
        # out_file = 'seg_local2global_%s_%d/%d.npz' % (
        #     merge_function, int(threshold*100), sub_block.block_id)
        # out_file = os.path.join(lut_dir, out_file)
        # np.savez_compressed(out_file, fragment_segment_lut=lut)
        dataset = 'seg_local2global_%s_%d' % (merge_function, int(threshold*100))
        lut = lut_db.save_lut(sub_block, lut, dataset=dataset)


def load_global_lut(threshold, lut_dir, merge_function, lut_filename=None):

    if lut_filename is None:
        lut_filename = 'seg_local2global_%s_%d_single' % (merge_function, int(threshold*100))
        # lut_filename = lut_filename + '_' + str(int(threshold*100))
    lut = os.path.join(
            lut_dir,
            lut_filename + '.npz')
    assert os.path.exists(lut), "%s does not exist" % lut
    start = time.time()
    logging.info("Reading global LUT...")
    lut = np.load(lut)['fragment_segment_lut']
    logging.info("%.3fs"%(time.time() - start))
    logging.info("Found %d fragments"%len(lut[0]))
    return lut


if __name__ == "__main__":

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
    block_size = daisy.Coordinate(block_size)

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)

    global_luts = {}
    for threshold in thresholds:
        global_luts[threshold] = load_global_lut(threshold, lut_dir, merge_function)

    while True:
        with daisy_client.acquire_block() as block:

            if block is None:
                break

            for threshold in thresholds:

                remap_in_block(
                    block,
                    block_size,
                    total_roi,
                    lut_dir,
                    merge_function,
                    threshold,
                    global_lut=global_luts[threshold],
                    chunk_size=tuple(chunk_size))

            # recording block done in the database
            completion_db.add_finished(block.block_id)
