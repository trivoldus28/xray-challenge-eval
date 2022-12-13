
import copy
import datetime
import os

import daisy
from daisy import Coordinate, Roi


def aggregateConfigs(configs):

    input_config = configs["Input"]
    global_config = configs.get("GlobalConfig", {})
    network_config = configs["Network"]
    # synapse_network_config = configs["SynfulNetwork"]

    if "xy_downsample" in input_config:
        print("`Input.xy_downsample` is deprecated!")
        print("Please use `Input.zyx_downsample`")
    if "xy_downsample" in network_config:
        print("`Network.xy_downsample` is deprecated!")
        print("Please use `Input.zyx_downsample`")

    today = datetime.date.today()
    parameters = {}
    parameters['year'] = today.year
    parameters['month'] = '%02d' % today.month
    parameters['day'] = '%02d' % today.day
    parameters['network'] = network_config['name']
    # parameters['synful_network'] = synapse_network_config['name']
    # parameters['synful_network1'] = synapse_network_config['name1']
    parameters['iteration'] = network_config['iteration']
    # parameters['synful_iteration'] = synapse_network_config['iteration']
    # parameters['synful_iteration1'] = synapse_network_config['iteration1']
    # parameters['raw_name'] = input_config['raw_name']
    config_filename = input_config['config_filename']

    parameters['proj'] = input_config.get('proj', '')
    if parameters['proj'] == '':
        # proj is just the last folder in the config path
        parameters['proj'] = config_filename.split('/')[-2]

    script_name = config_filename.split('/')[-1].split('.')
    if len(script_name) > 2:
        raise RuntimeError("script_name name %s cannot have more than two `.`")
    else:
        script_name = script_name[0]
    parameters['script_name'] = script_name
    parameters['script_folder'] = parameters['proj']
    parameters['script_dir'] = '/'.join(config_filename.split('/')[0:-1])
    script_dir = parameters['script_dir']

    input_config["experiment"] = input_config["experiment"].format(**parameters)
    parameters['experiment'] = input_config["experiment"]

    # input_config["output_file"] = input_config["output_file"].format(**parameters)

    input_config_synful = copy.deepcopy(input_config)
    input_config_synful1 = copy.deepcopy(input_config)
    # parameters_synful = copy.deepcopy(parameters)
    # parameters_synful['network'] = parameters_synful['synful_network']
    # parameters_synful['iteration'] = parameters_synful['synful_iteration']
    # parameters_synful1 = copy.deepcopy(parameters)
    # parameters_synful1['network'] = parameters_synful1['synful_network1']
    # parameters_synful1['iteration'] = parameters_synful1['synful_iteration1']

    for config in input_config:
        if isinstance(input_config[config], str):
            input_config[config] = input_config[config].format(**parameters)

    # for config in input_config_synful:
    #     if isinstance(input_config_synful[config], str):
    #         input_config_synful[config] = input_config_synful[config].format(**parameters_synful)
    # for config in input_config_synful1:
    #     if isinstance(input_config_synful1[config], str):
    #         input_config_synful1[config] = input_config_synful1[config].format(**parameters_synful1)

    configs["output_file"] = input_config["output_file"]
    configs["synful_output_file"] = input_config_synful["output_file"]
    configs["synful_output_file1"] = input_config_synful1["output_file"]

    for path_name in ["output_file", "synful_output_file", "synful_output_file1"]:

        output_path = configs[path_name]
        if not os.path.exists(output_path):
            output_path = os.path.join(script_dir, output_path)
        output_path = os.path.abspath(output_path)
        if output_path.startswith("/mnt/orchestra_nfs/"):
            output_path = output_path[len("/mnt/orchestra_nfs/"):]
            output_path = "/n/groups/htem/" + output_path

    os.makedirs(input_config['log_dir'], exist_ok=True)

    merge_function = configs["AgglomerateTask"]["merge_function"]
    thresholds_lut = configs["GlobalConfig"]["thresholds"]

    voxel_size = Coordinate(configs["Input"]["voxel_size"])
    def mult_voxel(config, key):
        if key in config:
            config[key] = Coordinate(config[key])
            config[key] *= voxel_size

    if "Input" in configs:
        config = configs["Input"]
        if config.get("size_in_pix", False):
            mult_voxel(config, "sub_roi_offset")
            mult_voxel(config, "sub_roi_shape")
            mult_voxel(config, "roi_offset")
            mult_voxel(config, "roi_shape")
            mult_voxel(config, "roi_context")

    for config in configs:

        if "Task" not in config:
            # print("Skipping %s" % config)
            continue

        config = configs[config]
        copyParameter(input_config, config, 'db_name')
        copyParameter(input_config, config, 'db_host')
        copyParameter(input_config, config, 'log_dir')
        copyParameter(input_config, config, 'sub_roi_offset')
        copyParameter(input_config, config, 'sub_roi_shape')

        if 'num_workers' in config:
            config['num_workers'] = int(config['num_workers'])

    if "GlobalConfig" in configs:
        config = configs["GlobalConfig"]
        if config.get("block_size_in_pix", False):
            mult_voxel(config, "fragments_block_size")
            mult_voxel(config, "fragments_context")
            mult_voxel(config, "agglomerate_block_size")
            mult_voxel(config, "agglomerate_context")
            mult_voxel(config, "find_segments_block_size")
            mult_voxel(config, "write_size")

    if "PredictTask" in configs:
        config = configs["PredictTask"]
        config['raw_file'] = input_config['raw_file']
        config['raw_dataset'] = input_config['raw_dataset']
        if 'out_file' not in config:
            config['out_file'] = input_config['output_file']
        config['train_dir'] = network_config['train_dir']
        config['iteration'] = network_config['iteration']
        copyParameter(network_config, config, 'net_voxel_size')
        copyParameter(network_config, config, 'predict_file')
        if 'predict_file' not in config or config['predict_file'] is None:
            config['predict_file'] = "predict.py"
        copyParameter(input_config, config, 'zyx_downsample')
        if 'roi_offset' in input_config:
            config['roi_offset'] = input_config['roi_offset']
        if 'roi_shape' in input_config:
            config['roi_shape'] = input_config['roi_shape']
        if 'roi_context' in input_config:
            config['roi_context'] = input_config['roi_context']
        copyParameter(input_config, config, 'delete_section_list')
        copyParameter(input_config, config, 'replace_section_list')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        copyParameter(input_config, config, 'center_roi_offset')
        copyParameter(input_config, config, 'overwrite')
        copyParameter(input_config, config, 'roi_shrink_context')
        copyParameter(input_config, config, 'roi_context')
        copyParameter(network_config, config, 'output_key')

    if "FixRawFromCatmaidTask" in configs:
        config = configs["FixRawFromCatmaidTask"]
        copyParameter(input_config, config, 'raw_file')
        copyParameter(input_config, config, 'raw_dataset')

    # if "PredictMyelinTask" in configs:
    #     raise RuntimeError("Deprecated task")
    #     config = configs["PredictMyelinTask"]
    #     config['raw_file'] = input_config['raw_file']
    #     config['myelin_file'] = input_config['output_file']
    #     if 'roi_offset' in input_config:
    #         config['roi_offset'] = input_config['roi_offset']
    #     if 'roi_shape' in input_config:
    #         config['roi_shape'] = input_config['roi_shape']

    # if "PredictCapillaryTask" in configs:
    #     config = configs["PredictCapillaryTask"]
    #     config['raw_file'] = input_config['raw_file']
    #     copyParameter(input_config, config, 'raw_dataset')
    #     config['out_file'] = input_config['output_file']
    #     if 'roi_offset' in input_config:
    #         config['roi_offset'] = input_config['roi_offset']
    #     if 'roi_shape' in input_config:
    #         config['roi_shape'] = input_config['roi_shape']
    #     copyParameter(input_config, config, 'replace_section_list')

    # if "MergeMyelinTask" in configs:
    #     config = configs["MergeMyelinTask"]
    #     if 'affs_file' not in config:
    #         config['affs_file'] = input_config['output_file']
    #     config['myelin_file'] = input_config['output_file']
    #     config['merged_affs_file'] = input_config['output_file']
    #     config['log_dir'] = input_config['log_dir']

    if "DownsampleTask" in configs:
        config = configs["DownsampleTask"]
        copyParameter(input_config, config, 'output_file', 'affs_file')

    if "ExtractFragmentTask" in configs:
        config = configs["ExtractFragmentTask"]
        copyParameter(input_config, config, 'output_file', 'affs_file')
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        # copyParameter(input_config, config, 'output_file', 'capillary_pred_file')
        copyParameter(input_config, config, 'raw_file')
        copyParameter(input_config, config, 'raw_dataset')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        copyParameter(input_config, config, 'db_file_name')
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'fragments_block_size', 'block_size')
        copyParameter(global_config, config, 'fragments_context', 'context')

    if "AgglomerateTask" in configs:
        config = configs["AgglomerateTask"]
        if 'affs_file' not in config:
            config['affs_file'] = input_config['output_file']
        config['fragments_file'] = input_config['output_file']
        config['merge_function'] = merge_function
        copyParameter(input_config, config, 'sub_roi_offset')
        copyParameter(input_config, config, 'sub_roi_shape')
        copyParameter(input_config, config, 'overwrite_sections')
        copyParameter(input_config, config, 'overwrite_mask_f')
        config['edges_collection'] = "edges_" + merge_function
        copyParameter(input_config, config, 'db_file_name')
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'fragments_block_size', 'filedb_nodes_chunk_size')
        copyParameter(global_config, config, 'agglomerate_block_size', 'block_size')
        copyParameter(global_config, config, 'agglomerate_block_size', 'filedb_edges_chunk_size')
        copyParameter(global_config, config, 'agglomerate_context', 'context')

    if "FindSegmentsGetLocalLUTsTask" in configs:
        config = configs["FindSegmentsGetLocalLUTsTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        config['edges_collection'] = "edges_" + merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'db_file_name')
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config,
                      'fragments_block_size', 'filedb_nodes_chunk_size')
        copyParameter(global_config, config,
                      'agglomerate_block_size', 'filedb_edges_chunk_size')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    # if "MakeInterThresholdMappingTask" in configs:
    #     config = configs["MakeInterThresholdMappingTask"]
    #     copyParameter(input_config, config, 'output_file', 'fragments_file')
    #     config['merge_function'] = merge_function
    #     config['edges_collection'] = "edges_" + merge_function

    if "FindSegmentsGetLocalEdgesTask" in configs:
        config = configs["FindSegmentsGetLocalEdgesTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    # if "FindSegmentsBlockwiseTask2a" in configs:
    #     config = configs["FindSegmentsBlockwiseTask2a"]
    #     copyParameter(input_config, config, 'output_file', 'fragments_file')
    #     config['merge_function'] = merge_function
    #     if 'thresholds' not in config:
    #         config['thresholds'] = thresholds_lut
    #     copyParameter(input_config, config, 'overwrite')
    #     copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    # if "FindSegmentsBlockwiseTask2b" in configs:
    #     config = configs["FindSegmentsBlockwiseTask2b"]
    #     copyParameter(input_config, config, 'output_file', 'fragments_file')
    #     config['merge_function'] = merge_function
    #     if 'thresholds' not in config:
    #         config['thresholds'] = thresholds_lut
    #     copyParameter(input_config, config, 'overwrite')
    #     copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    if "FindSegmentsGetGlobalLUTsTask" in configs:
        config = configs["FindSegmentsGetGlobalLUTsTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    if "FindSegmentsGetChunkedGlobalLUTsTask" in configs:
        config = configs["FindSegmentsGetChunkedGlobalLUTsTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    if "ExtractSegmentationTask" in configs:
        config = configs["ExtractSegmentationTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        copyParameter(input_config, config, 'output_file', 'out_file')
        config['merge_function'] = merge_function
        if 'thresholds' not in config:
            config['thresholds'] = thresholds_lut
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')
        copyParameter(global_config, config, 'write_size')

    if "ExtractSuperFragmentSegmentationTask" in configs:
        config = configs["ExtractSuperFragmentSegmentationTask"]
        copyParameter(input_config, config, 'output_file', 'fragments_file')
        copyParameter(input_config, config, 'output_file', 'out_file')
        config['merge_function'] = merge_function
        copyParameter(input_config, config, 'overwrite')
        copyParameter(global_config, config, 'find_segments_block_size', 'block_size')

    # network_config = configs["SynfulNetwork"]

    # if "PredictSynapseTask" in configs:
    #     config = configs["PredictSynapseTask"]
    #     os.makedirs(input_config_synful['log_dir'], exist_ok=True)
    #     # print(input_config_synful); exit()
    #     config['raw_file'] = input_config_synful['raw_file']
    #     config['raw_dataset'] = input_config_synful['raw_dataset']
    #     if 'out_file' not in config:
    #         config['out_file'] = input_config_synful['output_file']
    #     copyParameter(network_config, config, 'train_dir')
    #     copyParameter(network_config, config, 'iteration')
    #     config['log_dir'] = input_config_synful['log_dir']
    #     copyParameter(network_config, config, 'net_voxel_size')
    #     config['predict_file'] = network_config.get(
    #         'predict_file', 'segway/synful_tasks/predict.py')
    #     copyParameter(network_config, config, 'predict_file')
    #     copyParameter(network_config, config, 'xy_downsample')
    #     copyParameter(network_config, config, 'zyx_downsample')
    #     copyParameter(input_config_synful, config, 'roi_offset')
    #     copyParameter(input_config_synful, config, 'roi_shape')
    #     copyParameter(input_config_synful, config, 'sub_roi_offset')
    #     copyParameter(input_config_synful, config, 'sub_roi_shape')
    #     copyParameter(input_config_synful, config, 'delete_section_list')
    #     copyParameter(input_config_synful, config, 'replace_section_list')
    #     copyParameter(input_config_synful, config, 'overwrite_sections')
    #     copyParameter(input_config_synful, config, 'overwrite_mask_f')
    #     copyParameter(input_config_synful, config, 'center_roi_offset')
    #     copyParameter(network_config, config, 'out_properties')
    #     copyParameter(input_config, config, 'overwrite')

    # if "ExtractSynapsesTask" in configs:
    #     config = configs["ExtractSynapsesTask"]
    #     # config['raw_file'] = input_config['raw_file']
    #     # config['raw_dataset'] = input_config['raw_dataset']
    #     copyParameter(input_config, config, 'sub_roi_offset')
    #     copyParameter(input_config, config, 'sub_roi_shape')
    #     copyParameter(input_config, config, 'output_file', 'super_fragments_file')
    #     copyParameter(input_config, config, 'output_file', 'syn_indicator_file')
    #     copyParameter(input_config, config, 'output_file', 'syn_dir_file')
    #     copyParameter(input_config, config, 'overwrite')

    # if "PredictSynapseDirTask" in configs:
    #     config = configs["PredictSynapseDirTask"]
    #     os.makedirs(input_config_synful1['log_dir'], exist_ok=True)
    #     config['raw_file'] = input_config_synful1['raw_file']
    #     config['raw_dataset'] = input_config_synful1['raw_dataset']
    #     if 'out_file' not in config:
    #         config['out_file'] = input_config_synful1['output_file']
    #     copyParameter(network_config, config, 'train_dir1', 'train_dir')
    #     copyParameter(network_config, config, 'iteration1', 'iteration')
    #     copyParameter(network_config, config, 'out_properties1', 'out_properties')
    #     config['log_dir'] = input_config_synful1['log_dir']
    #     copyParameter(network_config, config, 'net_voxel_size')
    #     config['predict_file'] = 'segway/synful_tasks/predict.py'
    #     copyParameter(network_config, config, 'predict_file')
    #     copyParameter(network_config, config, 'xy_downsample')
    #     copyParameter(network_config, config, 'zyx_downsample')
    #     copyParameter(input_config_synful1, config, 'roi_offset')
    #     copyParameter(input_config_synful1, config, 'roi_shape')
    #     copyParameter(input_config_synful1, config, 'sub_roi_offset')
    #     copyParameter(input_config_synful1, config, 'sub_roi_shape')
    #     copyParameter(input_config_synful1, config, 'delete_section_list')
    #     copyParameter(input_config_synful1, config, 'replace_section_list')
    #     copyParameter(input_config_synful1, config, 'overwrite_sections')
    #     copyParameter(input_config_synful1, config, 'overwrite_mask_f')
    #     copyParameter(input_config_synful1, config, 'center_roi_offset')
    #     copyParameter(input_config, config, 'overwrite')


def copyParameter(from_config, to_config, name, to_name=None):

    if to_name is None:
        to_name = name
    if name in from_config and to_name not in to_config:
        to_config[to_name] = from_config[name]
