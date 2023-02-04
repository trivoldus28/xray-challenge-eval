import os.path
import math
import networkx as nx
import numpy as np
import argparse
from itertools import combinations, product
from collections import defaultdict

import daisy
import funlib.evaluate


def generate_graphs_with_seg_labels(segment_array, skeleton_path):
    '''
    Add predicted labels to the ground-truth graph. We also re-assign unique ids to
    each cluster of connected nodes after removal of nodes outside of the ROI.
    This is to differentiate between sets of nodes that are discontinuous in
    the ROI but originally belonged to the same skeleton ID
    '''
    gt_graph = np.load(skeleton_path, allow_pickle=True)
    next_highest_seg_label = int(segment_array.data.max()) + 1
    nodes_outside_roi = []
    for i, (treenode, attr) in enumerate(gt_graph.nodes(data=True)):
        pos = attr["position"]
        attr['zyx_coord'] = (pos[2], pos[1], pos[0])
        try:
            attr['seg_label'] = segment_array[daisy.Coordinate(attr['zyx_coord'])]
        except AssertionError as e:
            nodes_outside_roi.append(treenode)
            continue
        if attr['seg_label'] == 0:
            # We'll need to relabel them to be a unique non-zero value
            # for the Rand/VOI function to work. We also count contiguous skeletal
            # nodes predicted to be 0 as split errors.
            attr['seg_label'] = next_highest_seg_label
            set_point_in_array(array=segment_array, point_coord=attr['zyx_coord'],
                               val=next_highest_seg_label)
            next_highest_seg_label += 1

    for node in nodes_outside_roi:
        gt_graph.remove_node(node)

    # reassign `skeleton_id` attribute used in eval functions
    skel_clusters = nx.connected_components(gt_graph)
    for i, cluster in enumerate(skel_clusters):
        for node in cluster:
            gt_graph.nodes[node]['skeleton_id'] = i
    return gt_graph


def eval_erl(graph):
    node_seg_lut = {}
    for node, attr in graph.nodes(data=True):
        node_seg_lut[node] = attr['seg_label']

    # get total skel length
    skeleton_lengths = funlib.evaluate.run_length.get_skeleton_lengths(
        skeletons=graph,
        skeleton_position_attributes=['zyx_coord'],
        skeleton_id_attribute='skeleton_id')
    skeleton_lengths = [l for _, l in skeleton_lengths.items() if l > 0]
    average_skel_length = np.mean(skeleton_lengths)

    erl = funlib.evaluate.expected_run_length(
        skeletons=graph, skeleton_id_attribute='skeleton_id',
        node_segment_lut=node_seg_lut, skeleton_position_attributes=['zyx_coord'],
        return_merge_split_stats=False, edge_length_attribute='edge_length')
    erl_norm = erl/average_skel_length

    return erl, erl_norm


def build_segment_label_subgraph(segment_nodes, graph):
    subgraph = graph.subgraph(segment_nodes)
    skeleton_clusters = nx.connected_components(subgraph)
    seg_graph = nx.Graph()
    seg_graph.add_nodes_from(subgraph.nodes)
    seg_graph.add_edges_from(subgraph.edges)
    for skeleton_1, skeleton_2 in combinations(skeleton_clusters, 2):
        try:
            node_1 = skeleton_1.pop()
            node_2 = skeleton_2.pop()
            if graph.nodes[node_1]['skeleton_id'] == graph.nodes[node_2]['skeleton_id']:
                seg_graph.add_edge(node_1, node_2)
        except KeyError:
            pass
    return seg_graph


# Returns the closest pair of nodes on 2 skeletons
def get_closest_node_pair_between_two_skeletons(skel1, skel2, graph):
    multiplier = (1, 1, 1)
    shortest_len = math.inf
    for node1, node2 in product(skel1, skel2):
        coord1, coord2 = graph.nodes[node1]['zyx_coord'], graph.nodes[node2]['zyx_coord']
        distance = math.sqrt(sum([(a-b)**2 for a, b in zip(coord1, coord2)]))
        if distance < shortest_len:
            shortest_len = distance
            edge_attributes = {'distance': shortest_len}
            closest_pair = (node1, node2, edge_attributes)
    return closest_pair


def find_merge_errors(graph):
    seg_dict = {}
    for nid, attr in graph.nodes(data=True):
        seg_label = attr['seg_label']
        assert seg_label != 0, "Processed predicted labels cannot be 0"
        try:
            seg_dict[seg_label].add(nid)
        except KeyError:
            seg_dict[seg_label] = {nid}

    merge_errors = set()
    for seg_label, nodes in seg_dict.items():
        seg_graph = build_segment_label_subgraph(nodes, graph)
        skel_clusters = list(nx.connected_components(seg_graph))
        if len(skel_clusters) <= 1:
            continue
        potential_merge_sites = []
        for skeleton_1, skeleton_2 in combinations(skel_clusters, 2):
            shortest_connection = get_closest_node_pair_between_two_skeletons(
                                  skeleton_1, skeleton_2, graph)
            potential_merge_sites.append(shortest_connection)

        merge_sites = [(error_site[0], error_site[1]) for error_site in potential_merge_sites]
        merge_errors |= set(merge_sites)

    return merge_errors


def get_split_merges(graph):
    # Count split errors. An error is an edge in the GT skeleton graph connecting two nodes
    # of different segment ids.
    split_errors = []
    for edge in graph.edges():
        if graph.nodes[edge[0]]['seg_label'] != graph.nodes[edge[1]]['seg_label']:
            split_errors.append(edge)
    merge_errors = find_merge_errors(graph)
    return split_errors, merge_errors


def set_point_in_array(array, point_coord, val):
    '''Helper function to set value using real-world nm coordinates'''
    point_coord = daisy.Coordinate(point_coord)
    vox_aligned_offset = (point_coord // array.voxel_size) * array.voxel_size
    point_roi = daisy.Roi(vox_aligned_offset, array.voxel_size)
    array[point_roi] = val


def make_voxel_gt_array(test_array, gt_graph):
    '''Rasterize GT points to an empty array to compute Rand/VOI'''
    gt_ndarray = np.zeros_like(test_array.data)
    gt_array = daisy.Array(gt_ndarray, roi=test_array.roi, voxel_size=test_array.voxel_size)
    for neuron_id, cluster in enumerate(nx.connected_components(gt_graph)):
        for point in cluster:
            point_coord = gt_graph.nodes[point]['zyx_coord']
            set_point_in_array(array=gt_array, point_coord=point_coord, val=neuron_id)
    return gt_array


def get_voi(segment_array, gt_graph):
    '''Wrapper fn to compute Rand/VOI scores'''
    voxel_gt = make_voxel_gt_array(segment_array, gt_graph)
    res = funlib.evaluate.rand_voi(truth=voxel_gt.data, test=segment_array.data)
    return res


def run_eval(skeleton_file, segmentation_file, segmentation_ds, roi, downsampling=None):

    # load segmentation
    segment_array = daisy.open_ds(segmentation_file, segmentation_ds)
    if roi is None:
        roi = segment_array.roi
    if not segment_array.roi.contains(roi):
        raise RuntimeError(f"Provided segmentation ROI ({segment_array.roi}) does not contain test ROI ({roi})")
    segment_array = segment_array[roi]

    if downsampling is not None:
        assert type(downsampling) == int
        ndarray = segment_array.data[::downsampling, ::downsampling, ::downsampling]
        ds_voxel_size = segment_array.voxel_size * downsampling
        # align ROI
        roi_begin = (segment_array.roi.begin // ds_voxel_size) * ds_voxel_size
        roi_shape = daisy.Coordinate(ndarray.shape) * ds_voxel_size
        segment_array = daisy.Array(data=ndarray,
                                    roi=daisy.Roi(roi_begin, roi_shape),
                                    voxel_size=ds_voxel_size)

    # load to mem
    segment_array.materialize()
    voxel_size = segment_array.voxel_size
    ret = {}

    # compute GT graph
    gt_graph = generate_graphs_with_seg_labels(segment_array, skeleton_file)
    n_neurons = len(list(nx.connected_components(gt_graph)))
    ret['n_neurons'] = n_neurons
    ret['gt_graph'] = gt_graph

    # Compute ERL
    ret['erl'], ret['erl_norm'] = eval_erl(gt_graph)

    # Compute merge-split
    split_edges, merged_edges = get_split_merges(gt_graph)
    ret['split_edges'] = split_edges
    ret['merged_edges'] = merged_edges

    # Compute Rand/VOI
    ret['rand_voi'] = get_voi(segment_array, gt_graph)

    # Compute xpress score
    # xpress score is 50% normed erl and 50% normed voi
    rand_voi = ret['rand_voi']
    ret['xpress_voi'] = 1-0.5*(rand_voi['nvi_split']+rand_voi['nvi_merge'])
    ret['xpress_rand'] = 0.5*(rand_voi['rand_split']+rand_voi['rand_merge'])
    ret['xpress_erl_voi'] = 0.5*ret['xpress_voi'] + 0.5*ret['erl_norm']
    ret['xpress_erl_rand'] = 0.5*ret['xpress_rand'] + 0.5*ret['erl_norm']

    return ret

def check_submission_correctness(segmentation_file,
                                 check_ds_name='submission',
                                 check_resolution=(99,99,99),
                                 check_roi=daisy.Roi((3267,3267,3267), (33066,33066,33066)),
                                 ):
    try:
        segment_array = daisy.open_ds(segmentation_file, check_ds_name)
    except:
        raise RuntimeError(f'Fail: cannot open {segmentation_file} with dataset {check_ds_name}')
    assert segment_array.voxel_size == check_resolution, f"Fail: voxel_size is not {check_resolution}"
    assert segment_array.roi == check_roi, f"Fail: roi is not {check_roi}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("segmentation_file", type=str, help='Input segmentation file')
    ap.add_argument("--segmentation_ds", type=str, default='submission', help='Input segmentation dataset')
    ap.add_argument("--skeleton_file", type=str, default=None, help='Coordinates have to be XYZ in nm')
    ap.add_argument("--roi_begin", type=str, default=None, help='In nm')
    ap.add_argument("--roi_shape", type=str, default=None, help='In nm')
    ap.add_argument("--downsampling", type=int, default=None, help='Downsampling factor to perform before evaluation')
    ap.add_argument("--print_errors", type=int, default=0, help='Print split/merge errors coordinates')
    ap.add_argument("--print_in_xyz", type=int, default=0, help='Print coordinates in xyz instead of zyx')
    ap.add_argument("--check_submission", type=int, default=0, help='Check correctness of xpress submission')
    ap.add_argument("--show_all_scores", type=int, default=0, help='')
    ap.add_argument("--mode", type=str, default=None, help='')
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    if check_submission:
        check_submission_correctness(segmentation_file)
        print('Pass')
        exit()

    if mode == "validation":
        roi_begin = '8316,8316,8316'
        roi_shape = '23067,23067,23067'
    elif mode == "test":
        roi_begin = '3267,3267,3267'
        roi_shape = '33066,33066,33066'

    assert skeleton_file is not None, "Error: skeleton_file is not given"

    # Parse ROI
    roi = None
    if roi_begin is not None or roi_begin is not None:
        roi_begin = [float(k) for k in roi_begin.split(',')]
        roi_shape = [float(k) for k in roi_shape.split(',')]
        roi = daisy.Roi(roi_begin, roi_shape)

    # Run eval fn
    res = run_eval(skeleton_file, segmentation_file, segmentation_ds, roi, downsampling)

    if not show_all_scores:
        # Print just the single xpress score
        print(f"{res['xpress_erl_rand']}")
        exit()

    # Terminal outputs
    print(f'n_neurons: {res["n_neurons"]}')
    print(f'Expected run-length: {res["erl"]}')
    print(f'Normalized ERL: {res["erl_norm"]}')

    split_edges = res['split_edges']
    merged_edges = res['merged_edges']
    print('Count results:')
    print(f'\tSplit count (total, per-neuron): {len(split_edges)}, {len(split_edges)/res["n_neurons"]}')
    print(f'\tMerge count (total, per-neuron): {len(merged_edges)}, {len(merged_edges)/res["n_neurons"]}')
    if print_errors:
        gt_graph = res['gt_graph']
        def print_coords(node1, node2):
            node1_coord = (daisy.Coordinate(gt_graph.nodes[node1]['zyx_coord']) / 33)
            node2_coord = (daisy.Coordinate(gt_graph.nodes[node2]['zyx_coord']) / 33)
            if print_in_xyz:
                node1_coord = node1_coord[::-1]
                node2_coord = node2_coord[::-1]
            print(f"{node1_coord} to {node2_coord}")

        print("Split errors:")
        splits_by_skel = defaultdict(list)
        for edge in split_edges:
            splits_by_skel[gt_graph.nodes[edge[0]]['skeleton_id']].append(edge)
        for skel in splits_by_skel:
            print(f'Skeleton #{skel}')
            for edge in splits_by_skel[skel]:
                print_coords(edge[0], edge[1])
        print("Split error histogram:")
        split_histogram = defaultdict(int)
        for i in range(res["n_neurons"]):
            split_histogram[len(splits_by_skel[i])] += 1
        for k in sorted(split_histogram):
            print(f'{k}: {split_histogram[k]}')

        print("Merge errors:")
        for node1, node2 in merged_edges:
            print_coords(node1, node2)

    rand_voi = res['rand_voi']
    print("Rand results (higher better):")
    print(f"\tRand split: {rand_voi['rand_split']}")
    print(f"\tRand merge: {rand_voi['rand_merge']}")
    print("VOI results (lower better):")
    print(f"\tNormalized VOI split: {rand_voi['nvi_split']}")
    print(f"\tNormalized VOI merge: {rand_voi['nvi_merge']}")

    print("XPRESS score (higher is better):")
    print(f"\tERL+VOI : {res['xpress_erl_voi']}")
    print(f"\tERL+RAND: {res['xpress_erl_rand']}")
    print(f"\tVOI     : {res['xpress_voi']}")
    print(f"\tRAND    : {res['xpress_rand']}")

