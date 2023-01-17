import json
import logging
import sys
import time
import os
import numpy as np

import daisy

from funlib.segment.graphs.impl import connected_components

from lut import LookupTable
from base_task import Database

logger = logging.getLogger(__name__)
# logging.getLogger('daisy').setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
# logging.getLogger('lut').setLevel(logging.DEBUG)


def read_block(graph_provider, block):

    logger.debug("Reading graph in block %s", block)
    start = time.time()
    graph = graph_provider[block.read_roi]
    logger.debug("Read graph from graph provider in %.3fs",
                 time.time() - start)

    nodes = {
        'id': []
    }
    edges = {
        'u': [],
        'v': []
    }

    start = time.time()
    for node, data in graph.nodes(data=True):

        # skip over nodes that are not part of this block (they have been
        # pulled in by edges leaving this block and don't have a position
        # attribute)

        if type(graph_provider.position_attribute) == list:
            probe = graph_provider.position_attribute[0]
        else:
            probe = graph_provider.position_attribute
        if probe not in data:
            continue

        nodes['id'].append(np.uint64(node))
        for k, v in data.items():
            if k not in nodes:
                nodes[k] = []
            nodes[k].append(v)

    for u, v, data in graph.edges(data=True):

        edges['u'].append(np.uint64(u))
        edges['v'].append(np.uint64(v))
        for k, v in data.items():
            if k not in edges:
                edges[k] = []
            edges[k].append(v)

    if len(nodes['id']) == 0:
        logger.debug("Graph is empty")
        return

    if len(edges['u']) == 0:
        # no edges in graph, make sure empty np array has correct dtype
        edges['u'] = np.array(edges['u'], dtype=np.uint64)
        edges['v'] = np.array(edges['v'], dtype=np.uint64)

    nodes = {
        k: np.array(v)
        for k, v in nodes.items()
    }
    edges = {
        k: np.array(v)
        for k, v in edges.items()
    }
    logger.debug("Parsed graph in %.3fs", time.time() - start)

    # start = time.time()
    return (nodes, edges)


def find_segments(**kwargs):

    for key in kwargs:
        globals()['%s' % key] = kwargs[key]

    graph_provider = daisy.persistence.FileGraphProvider(
        directory=os.path.join(filedb_file, filedb_dataset),
        chunk_size=filedb_edges_chunk_size,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x'],
        save_attributes_as_single_file=True,
        nodes_roi_offset=filedb_nodes_roi_offset,
        edges_roi_offset=filedb_edges_roi_offset,
        nodes_chunk_size=filedb_nodes_chunk_size,
        edges_chunk_size=filedb_edges_chunk_size,
        nodes_no_misaligned_reads=True,
        # edges_no_misaligned_reads=True,
        )

    res = read_block(graph_provider, block)

    if res is None:
        if not ignore_degenerates:
            raise RuntimeError('No nodes found in %s' % block)
        else:
            logger.info('No nodes found in %s' % block)
        # create dummy nodes
        node_attrs = {
            'id': np.array([0], dtype=np.uint64),
        }
        edge_attrs = {
            'u': np.array([0]),
            'v': np.array([0]),
        }
    else:
        node_attrs, edge_attrs = res

    if 'id' not in node_attrs:
        if not ignore_degenerates:
            raise RuntimeError('No nodes found in %s' % block)
        else:
            logger.info('No nodes found in %s' % block)
        nodes = [0]
    else:
        nodes = node_attrs['id']

    u_array = edge_attrs['u'].astype(np.uint64)
    v_array = edge_attrs['v'].astype(np.uint64)
    edges = np.stack([u_array, v_array], axis=1)

    logger.info("RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    if len(u_array) == 0:
        # this block somehow has no edges, or is not agglomerated
        u_array = np.array([0], dtype=np.uint64)
        v_array = np.array([0], dtype=np.uint64)
        edges = np.array([[0, 0]], dtype=np.uint64)

        if not ignore_degenerates:
            # print error and abort
            print(f"ERROR: block {block} somehow has no edges, or is not agglomerated")
            return 1

    if 'merge_score' in edge_attrs:
        scores = edge_attrs['merge_score'].astype(np.float32)
    else:
        scores = np.ones_like(u_array, dtype=np.float32)

    # each block should have at least one node, edge, and score
    assert len(nodes)
    assert len(edges)
    assert len(scores)

    out_dir = os.path.join(
        fragments_file,
        lut_dir)

    for threshold in thresholds:
        get_connected_components(
                nodes,
                edges,
                scores,
                threshold,
                merge_function,
                out_dir,
                block,
                ignore_degenerates=ignore_degenerates)


def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        merge_function,
        out_dir,
        block,
        hi_threshold=0.95,
        ignore_degenerates=False,
        **kwargs):

    logger.debug("Getting CCs for threshold %.3f..." % threshold)

    edges_tmp = edges[scores <= threshold]
    scores_tmp = scores[scores <= threshold]

    if len(edges_tmp):
        components = connected_components(nodes, edges_tmp, scores_tmp, threshold,
                                          use_node_id_as_component_id=1)

    else:
        if len(scores) == 0:
            print("edges_tmp: ", edges_tmp)
            print("scores_tmp: ", scores_tmp)
            print("edges: ", edges)
            print("scores: ", scores)
            print("len(nodes): ", len(nodes))
            if not ignore_degenerates:
                raise RuntimeError(
                    'Empty edges in graph! Likely unfinished agglomeration.')
            else:
                logger.info(
                    'Empty edges in graph! Likely unfinished agglomeration.')
        components = nodes

    lut = np.array([nodes, components])
    lut_db = LookupTable(filepath=out_dir)

    dataset = 'seg_frags2local_%s_%d' % (merge_function, int(threshold*100))
    lut_db.save_lut(block, lut, dataset=dataset)

    unique_components = np.unique(components)
    dataset = 'nodes_%s_%d' % (merge_function, int(threshold*100))
    lut_db.save_nodes(block, unique_components, dataset=dataset)

    nodes_in_vol = set(nodes)

    def not_in_graph(u, v):
        return u not in nodes_in_vol or v not in nodes_in_vol

    logger.debug(f"Num edges original: {len(edges)}")
    outward_edges = np.array([not_in_graph(n[0], n[1]) for n in edges])
    edges = edges[np.logical_and(scores <= threshold, outward_edges)]

    # replace IDs in edges with agglomerated IDs
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

    logger.debug(f"Num edges pruned: {len(edges)}")

    dataset = 'edges_local2frags_%s_%d' % (merge_function, int(threshold*100))
    lut_db.save_edges(block, edges, dataset=dataset)


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

    while True:

        with daisy_client.acquire_block() as block:

            if block is None:
                break

            logger.info("Block: %s" % block)

            find_segments(
                db_host=db_host,
                db_name=db_name,
                filedb_file=filedb_file,
                filedb_dataset=filedb_dataset,
                fragments_file=fragments_file,
                lut_dir=lut_dir,
                edges_collection=edges_collection,
                merge_function=merge_function,
                block=block,
                thresholds=thresholds,
                ignore_degenerates=ignore_degenerates,
                )

            # recording block done in the database
            completion_db.add_finished(block.block_id)
