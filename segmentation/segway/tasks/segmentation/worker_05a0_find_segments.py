import daisy
import json
import logging
import sys
import time
import os
import glob

import numpy as np

from funlib.segment.graphs.impl import connected_components
from base_task import Database

# logging.basicConfig(level=logging.INFO)
# np.set_printoptions(threshold=sys.maxsize, formatter={'all':lambda x: str(x)})

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
# logging.getLogger('lut').setLevel(logging.DEBUG)


def find_segments(
        fragments_file,
        lut_dir,
        merge_function,
        # roi_offset,
        # roi_shape,
        # total_roi,
        thresholds,
        **kwargs):
    '''Computing the global connected component LUTs.
    This step is only necessary for computing whole-dataset segmentation
    (as opposed to a block-wise segmentation).'''

    lut_dir = os.path.join(
        fragments_file,
        lut_dir)

    start = time.time()

    for threshold in thresholds:

        # load edges
        logger.debug("Loading edges_local2local...")
        start = time.time()
        edge_files_pat = 'edges_local2local_%s_%d/*/*/*.npz' % (merge_function, int(threshold*100))
        edge_files = glob.glob(lut_dir + '/' + edge_files_pat)
        edges = []
        for edge_file in edge_files:
            edges.append(np.load(edge_file)['edges'])
        # logger.debug(edges)
        if len(edges) > 1:
            edges = np.concatenate(edges)
        else:
            edges = edges[0]
        logger.debug("%.3fs" % (time.time() - start))
        # logger.debug(edges)
        if len(edges) == 0:
            edges = np.array([[0, 0]], dtype=np.uint64)

        # load nodes
        logger.debug("Loading nodes...")
        start = time.time()
        node_files_pat = 'nodes_%s_%d/*/*/*.npz' % (merge_function, int(threshold*100))
        node_files = glob.glob(lut_dir + '/' + node_files_pat)
        nodes = []
        for node_file in node_files:
            nodes.append(np.load(node_file)['nodes'])
        # logger.debug(nodes)
        if len(nodes) > 1:
            nodes = np.concatenate(nodes)
        else:
            nodes = nodes[0]
        logger.debug("%.3fs" % (time.time() - start))

        logger.debug("Getting CCs...")
        start = time.time()
        components = connected_components(
                                    nodes, edges, scores=None, threshold=1.0,
                                    no_scores=1,
                                    use_node_id_as_component_id=1
                                    )
        logger.debug("%.3fs" % (time.time() - start))

        logger.debug("Creating fragment-segment LUT for threshold %.3f..." % threshold)
        start = time.time()
        lut = np.array([nodes, components])
        logger.debug("%.3fs" % (time.time() - start))

        logger.debug("Storing fragment-segment LUT for threshold %.3f..." % threshold)
        start = time.time()
        lookup = 'seg_local2global_%s_%d_single' % (merge_function, int(threshold*100))
        out_file = os.path.join(lut_dir, lookup)
        np.savez_compressed(out_file, fragment_segment_lut=lut)
        logger.debug("%.3fs" % (time.time() - start))


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

    while True:
        with daisy_client.acquire_block() as block:

            if block is None:
                break

            find_segments(
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                merge_function=merge_function,
                thresholds=thresholds,
                )

            # recording block done in the database
            completion_db.add_finished(block.block_id)

            daisy_client.release_block(block)
