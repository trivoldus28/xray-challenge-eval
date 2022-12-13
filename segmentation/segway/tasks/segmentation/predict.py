from __future__ import print_function
import sys
import json
import logging
import os
import glob
import pymongo

import gpu_utils
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_random_gpu_lowest_memory())
print("Running script on GPU #%s" % os.environ["CUDA_VISIBLE_DEVICES"])

from gunpowder import *
from gunpowder.contrib import ZeroOutConstSections
from gunpowder.tensorflow import *

from EditSectionsNode import ReplaceSectionsNode

def predict(
        iteration,
        raw_file,
        raw_dataset,
        voxel_size,
        out_file,
        out_dataset,
        train_dir,
        predict_num_core,
        zyx_downsample,
        config_file,
        meta_file,
        db_host,
        db_name,
        completion_db_name,
        delete_section_list=[],
        replace_section_list=[],
        ):

    setup_dir = train_dir

    with open(os.path.join(setup_dir, config_file), 'r') as f:
        net_config = json.load(f)

    # try to find checkpoint name
    pattern = '*checkpoint_%d.*' % iteration
    checkpoint_files = glob.glob(train_dir + '/' + pattern)
    if len(checkpoint_files) == 0:
        print("Cannot find checkpoints with pattern %s in directory %s" % (
            pattern, train_dir))
        os._exit(1)

    checkpoint_file = checkpoint_files[0].split('.')[0]
    checkpoint_file = checkpoint_file.split('/')[-1]

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

    outputs = {net_config['affs']: affs}
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
    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]
    completion_db = db[completion_db_name]

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

    print(zyx_downsample)
    if run_downsampling:
        pipeline += DownSample(rawfr, zyx_downsample, raw)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2, -1)

    pipeline += Predict(
            os.path.join(setup_dir, checkpoint_file),
            inputs={
                net_config['raw']: raw
            },
            outputs=outputs,
            graph=os.path.join(setup_dir, meta_file),
            # shared_memory_per_worker_MB=256,
            max_shared_memory=256*1024*1024,
            skip_empty=True,
        )
    pipeline += IntensityScaleShift(affs, 255, 0)

    pipeline += PrintProfilingStats(every=10)

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
            # worker_config,
            b, s, d)
        )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    db_client.close()
    print("Prediction finished")


def block_done_callback(
        completion_db,
        # worker_config,
        block,
        start,
        duration):

    pass
    document = dict()
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })
    completion_db.insert(document)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(
        run_config['iteration'],
        run_config['raw_file'],
        run_config['raw_dataset'],
        run_config['voxel_size'],
        run_config['out_file'],
        run_config['out_dataset'],
        run_config['train_dir'],
        run_config['predict_num_core'],
        run_config['zyx_downsample'],
        run_config['config_file'],
        run_config['meta_file'],
        run_config['db_host'],
        run_config['db_name'],
        run_config['completion_db_name'],
        run_config['delete_section_list'],
        run_config['replace_section_list'],
        )
