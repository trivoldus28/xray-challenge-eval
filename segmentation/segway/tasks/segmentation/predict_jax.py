from __future__ import print_function
import sys
import json
import logging
import os
import glob
import pymongo
import importlib

import gpu_utils
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_random_gpu_lowest_memory())
print("Running script on GPU #%s" % os.environ["CUDA_VISIBLE_DEVICES"])

import gunpowder as gp
from gunpowder import *
from gunpowder.contrib import ZeroOutConstSections
from EditSectionsNode import ReplaceSectionsNode
# from gunpowder.jax.nodes import Predict
from buffered_predict_jax_node import Predict

# from daisy_request_blocks import DaisyRequestBlocks
from base_task import Database


def predict(
        model_file,
        iteration,
        raw_file,
        raw_dataset,
        voxel_size,
        out_file,
        out_dataset,
        output_key,
        train_dir,
        predict_num_core,
        zyx_downsample,
        config_file,
        db_host,
        db_name,
        completion_db_name,
        delete_section_list=[],
        replace_section_list=[],
        ):

    sys.path.insert(0, train_dir)
    model = importlib.import_module(model_file).create_network()

    # with open(os.path.join(train_dir, config_file), 'r') as f:
    with open(config_file) as f:
        net_config = json.load(f)
        print(net_config)

    # try to find checkpoint name
    pattern = 'model_checkpoint_%d' % iteration
    checkpoint_files = glob.glob(train_dir + '/' + pattern)
    if len(checkpoint_files) == 0:
        print("Cannot find checkpoints with pattern %s in directory %s" % (
            pattern, train_dir))
        os._exit(1)

    checkpoint_file = checkpoint_files[0]

    # These values are in pixels/voxels
    input_shape = Coordinate(net_config["input_shape"])
    output_shape = Coordinate(net_config["output_shape"])
    voxel_size = Coordinate(tuple(voxel_size))

    context = (input_shape - output_shape)//2

    print("Context is %s"%(context,))
    input_size = input_shape*voxel_size
    output_size = output_shape*voxel_size

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    # outputs = {net_config['affs']: affs}
    dataset_names = {affs: out_dataset}

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    daisy_roi_map = {
        raw: "read_roi",
        affs: "write_roi"
    }

    run_downsampling = False
    if zyx_downsample is not None:
        run_downsampling = (zyx_downsample[0]*zyx_downsample[1]*zyx_downsample[2]) > 1
        zyx_downsample = tuple(zyx_downsample)

    if run_downsampling:
        rawfr = ArrayKey('RAWFR')
        chunk_request.add(rawfr, input_size)

    initial_raw = raw

    if run_downsampling:
        daisy_roi_map = {
            raw: "read_roi",
            rawfr: "read_roi",
            affs: "write_roi"
        }
        initial_raw = rawfr

    print("db_host: ", db_host)
    print("db_name: ", db_name)
    print("completion_db_name: ", completion_db_name)
    completion_db = Database(db_host, db_name, completion_db_name)

    if raw_file.endswith(".hdf"):
        pipeline = Hdf5Source(
            raw_file,
            datasets={initial_raw: raw_dataset},
            array_specs={initial_raw: ArraySpec(interpolatable=True)})
    elif raw_file.endswith(".zarr") or raw_file.endswith(".n5"):
        pipeline = ZarrSource(
            raw_file,
            datasets={initial_raw: raw_dataset},
            array_specs={initial_raw: ArraySpec(interpolatable=True)})
    else:
        raise RuntimeError("Unknown raw file type!")

    if len(delete_section_list) or len(replace_section_list):
        pipeline += ReplaceSectionsNode(
            initial_raw,
            delete_section_list=delete_section_list,
            replace_section_list=replace_section_list,
            )

    pipeline += Pad(initial_raw, size=None)

    if run_downsampling:
        print(zyx_downsample)
        pipeline += DownSample(rawfr, zyx_downsample, raw)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2, -1)

    # add "channel" dimensions
    pipeline += gp.Unsqueeze([raw])
    # add "batch" dimensions
    pipeline += gp.Unsqueeze([raw])

    pipeline += Predict(
            model=model,
            inputs={'raw': raw},
            # outputs={'affs': affs},
            outputs={output_key: affs},
            checkpoint=checkpoint_file,
            skip_empty=True,
            max_shared_memory=512*1024*1024,
            # spawn_subprocess=True,
            )

    pipeline += gp.Squeeze([affs], axis=0)
    # pipeline += gp.Squeeze([affs], axis=0)

    pipeline += IntensityScaleShift(affs, 255, 0)

    pipeline += PrintProfilingStats(every=100)

    pipeline += ZarrWrite(
            dataset_names=dataset_names,
            output_filename=out_file
        )

    pipeline += DaisyRequestBlocks(
        chunk_request,
        roi_map=daisy_roi_map,
        num_workers=predict_num_core,
        block_done_callback=lambda b, s, d: block_done_callback(
            completion_db,
            b, s, d)
        )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    db_client.close()
    print("Prediction finished")


def block_done_callback(
        completion_db,
        block,
        start,
        duration):
    # recording block done in the database
    completion_db.add_finished(block.block_id)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(
        run_config['model_file'],
        run_config['iteration'],
        run_config['raw_file'],
        run_config['raw_dataset'],
        run_config['voxel_size'],
        run_config['out_file'],
        run_config['out_dataset'],
        run_config['output_key'],
        run_config['train_dir'],
        run_config['predict_num_core'],
        run_config['zyx_downsample'],
        run_config['config_file'],
        run_config['db_host'],
        run_config['db_name'],
        run_config['completion_db_name'],
        run_config['delete_section_list'],
        run_config['replace_section_list'],
        )
