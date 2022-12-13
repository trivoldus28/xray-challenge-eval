import os.path
import networkx as nx
import numpy as np
import argparse

import daisy
from funlib import evaluate


def assign_skeleton_indexes(graph):
    '''Assign unique ids to each cluster of connected nodes. This is to
    differentiate between sets of nodes that are discontinuous in the
    ROI but actually belong to the same skeleton ID, which is necessary
    because the network should not be penalized for incorrectly judging
    that these processes belong to different neurons.'''
    skeleton_index_to_id = {}
    skel_clusters = nx.connected_components(graph)
    for i, cluster in enumerate(skel_clusters):
        for node in cluster:
            graph.nodes[node]['skeleton_index'] = i
        skeleton_index_to_id[i] = graph.nodes[cluster.pop()]['skeleton_id']
    return graph


def add_predicted_seg_labels_from_vol(
        graph, segmentation_file, segmentation_ds,
        load_segment_array_to_memory=True):

    bg_nodes=0
    segment_array = daisy.open_ds(segmentation_file, segmentation_ds)
    segment_array = segment_array[segment_array.roi]
    if load_segment_array_to_memory:
        segment_array.materialize()

    nodes_outside_roi = []  
    for i, (treenode, attr) in enumerate(graph.nodes(data=True)):
        pos = attr["position"]
        try:    
            attr['zyx_coord'] = (pos[2], pos[1], pos[0])
            attr['seg_label'] = segment_array[daisy.Coordinate(attr['zyx_coord'])]
            if attr['seg_label'] == 0:
                bg_nodes+=1
                raise AssertionError
        except AssertionError as e:
            nodes_outside_roi.append(treenode)

    print(f'Removing {len(nodes_outside_roi)} GT annotations outside of evaluated ROI')
    for node in nodes_outside_roi:
        graph.remove_node(node)

    print('BG_NODES',bg_nodes)
    return assign_skeleton_indexes(graph)


def generate_graphs_with_seg_labels(segmentation_file, segmentation_ds, skeleton_path, num_processes):
    unlabeled_skeleton = np.load(skeleton_path, allow_pickle=True)
    return add_predicted_seg_labels_from_vol(unlabeled_skeleton.copy(), segmentation_file, segmentation_ds)    


def eval_erl(skeleton_file, segmentation_file, segmentation_ds):
    
    node_seg_lut = {}
    graph_list = generate_graphs_with_seg_labels(segmentation_file, segmentation_ds, skeleton_file, 1)
    for node, attr in graph_list.nodes(data=True):
        node_seg_lut[node]=attr['seg_label']

    res = evaluate.expected_run_length(skeletons=graph_list,skeleton_id_attribute='skeleton_id',
                        node_segment_lut=node_seg_lut,skeleton_position_attributes=['zyx_coord'],
                        return_merge_split_stats=False,edge_length_attribute='edge_length')

    return res


if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("skeleton_file", type=str, help='Coordinates have to be XYZ in nm')
    ap.add_argument("segmentation_file", type=str, help='')
    ap.add_argument("segmentation_ds", type=str, help='')
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    res_erl = eval_erl(skeleton_file, segmentation_file, segmentation_ds)
    print(f'Expected run-length: {res_erl}')
