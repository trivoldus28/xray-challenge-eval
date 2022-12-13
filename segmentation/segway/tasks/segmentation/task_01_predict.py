import json
import logging
import os
import sys
import importlib
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate

import base_task
from base_task import Parameter
from batch_task import BatchTask

logger = logging.getLogger(__name__)


class PredictTask(BatchTask):
    '''Run prediction in parallel blocks. Within blocks, predict in chunks.
    Blank parameters are required.
    '''

    train_dir = Parameter()
    iteration = Parameter()
    raw_file = Parameter()
    raw_dataset = Parameter()
    lsds_file = Parameter(None)
    lsds_dataset = Parameter(None)
    mask_file = Parameter(None)
    mask_dataset = Parameter(None)

    out_file = Parameter()
    out_dataset = Parameter()
    block_size_in_chunks = Parameter([1, 1, 1])
    block_size_in_chunks_div = Parameter([1, 1, 1])
    num_workers = Parameter()
    predict_file = Parameter(None)

    roi_offset = Parameter(None)
    roi_shape = Parameter(None)
    roi_shrink_context = Parameter(False)
    roi_context = Parameter(None)
    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    net_voxel_size = Parameter(None)
    zyx_downsample = Parameter(None)

    num_cores_per_worker = Parameter(8)
    mem_per_core = Parameter(2)
    sbatch_gpu_type = Parameter('any')

    delete_section_list = Parameter([])
    replace_section_list = Parameter([])

    overwrite_mask_f = Parameter(None)
    overwrite_sections = Parameter(None)

    model_file = Parameter('mknet')
    net_config_file = Parameter(None)
    output_key = Parameter('affs')

    def prepare(self, upstream_tasks=None):

        self.setup = os.path.abspath(self.train_dir)
        self.raw_file = os.path.abspath(self.raw_file)
        self.out_file = os.path.abspath(self.out_file)

        logger.info('Input file path: ' + self.raw_file)
        logger.info('Output file path: ' + self.out_file)

        # from here on, all values are in world units (unless explicitly mentioned)

        # get ROI of source
        source = daisy.open_ds(self.raw_file, self.raw_dataset)
        logger.info("Source dataset has shape %s, ROI %s, voxel size %s"%(
            source.shape, source.roi, source.voxel_size))

        # load config
        if self.net_config_file is None:
            found = False
            for base in ['test_net', 'unet']:
                config_file = os.path.join(self.setup, base+'.json')
                meta_file = os.path.join(self.setup, base+'.meta')
                if os.path.exists(config_file):
                    found = True
                    break
            if not found:
                raise RuntimeError("No network config found at %s" % self.setup)
        else:
            config_file = self.net_config_file + '.json'
            meta_file = self.net_config_file + '.meta'

        net_config = json.load(open(config_file))

        input_shape = net_config["input_shape"]
        output_shape = net_config["output_shape"]

        if self.zyx_downsample is None:
            self.zyx_downsample = (1, 1, 1)
        elif isinstance(self.zyx_downsample, int):
            self.zyx_downsample = (self.zyx_downsample, self.zyx_downsample, self.zyx_downsample)
        self.zyx_downsample = Coordinate(self.zyx_downsample)

        # get chunk size and context
        voxel_size = source.voxel_size
        self.net_voxel_size = tuple(self.net_voxel_size)
        if self.net_voxel_size != source.voxel_size:
            assert source.voxel_size*self.zyx_downsample == self.net_voxel_size, (
                    f"Source voxel size {source.voxel_size} mult by "
                    f"downsample_factors {self.zyx_downsample} does not "
                    f"match network voxel size {self.net_voxel_size}")
            voxel_size = self.net_voxel_size

        net_input_size = Coordinate(input_shape)*Coordinate(voxel_size)
        net_output_size = Coordinate(output_shape)*Coordinate(voxel_size)
        chunk_size = net_output_size
        net_context = (net_input_size - net_output_size)/2

        # compute sizes of blocks
        if self.block_size_in_chunks is not None:
            logger.warn("block_size_in_chunks is deprecated")
        self.block_size_in_chunks = [1, 1, 1]
        block_output_size = chunk_size*Coordinate(self.block_size_in_chunks)
        block_input_size = block_output_size + net_context*2

        logger.info("Following sizes in world units:")
        logger.info("net input size  = %s" % (net_input_size,))
        logger.info("net output size = %s" % (net_output_size,))
        logger.info("net_context         = %s" % (net_context,))
        logger.info("chunk size      = %s" % (chunk_size,))

        # create read and write ROI
        block_read_roi = daisy.Roi((0, 0, 0), block_input_size) - net_context
        block_write_roi = daisy.Roi((0, 0, 0), block_output_size)

        if not self.roi_shrink_context:
            roi_context = net_context
        else:
            if self.roi_context is None:
                self.roi_context = net_context
            self.roi_context = Coordinate(self.roi_context)
        sched_roi, dataset_roi = base_task.compute_compatible_roi(
                roi_offset=self.roi_offset,
                roi_shape=self.roi_shape,
                sub_roi_offset=self.sub_roi_offset,
                sub_roi_shape=self.sub_roi_shape,
                roi_context=self.roi_context,
                net_context=net_context,
                chunk_size=chunk_size,
                source_roi=source.roi,
                voxel_size=voxel_size,
                shrink_context=self.roi_shrink_context,
            )

        logger.info("Following ROIs in world units:")
        logger.info("Total input ROI  = %s" % sched_roi)
        logger.info("Block read  ROI  = %s" % block_read_roi)
        logger.info("Block write ROI  = %s" % block_write_roi)
        logger.info("Total output ROI = %s" % dataset_roi)

        logging.info('Preparing output dataset')

        write_size = chunk_size
        write_size = write_size / Coordinate(self.block_size_in_chunks_div)

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        logger.info(f'affs dataset_roi: {dataset_roi}')
        logger.info(f'affs voxel_size: {voxel_size}')
        logger.info(f'affs write_size: {write_size}')

        try:
            # new format
            out_dims = net_config['outputs'][self.output_key]['out_dims']
            out_dtype = net_config['outputs'][self.output_key]['out_dtype']
        except:
            # try old format
            out_dims = net_config['out_dims']
            out_dtype = net_config['out_dtype']
        logger.info('Number of dimensions is %i' % out_dims)

        self.affs_ds = daisy.prepare_ds(
            self.out_file,
            self.out_dataset,
            dataset_roi,
            voxel_size,
            out_dtype,
            write_size=write_size,
            force_exact_write_size=True,
            num_channels=out_dims,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=delete_ds,
            )

        if self.raw_file.endswith('.json'):
            with open(self.raw_file, 'r') as f:
                spec = json.load(f)
                self.raw_file = spec['container']

        self.overwrite_mask = None
        if self.overwrite_mask_f:
            # force precheck = False for any ROI with any voxel in mask = 1
            self.overwrite_mask = daisy.open_ds(
                self.overwrite_mask_f, "overwrite_mask")

        if self.overwrite_sections is not None:
            write_shape = [k for k in dataset_roi.get_shape()]
            write_shape[0] = 40
            write_shape = tuple(write_shape)

            rois = []
            # overwrite_sections_begin = dataset_roi.get_begin()
            for s in self.overwrite_sections:
                write_offset = [k for k in dataset_roi.get_begin()]
                write_offset[0] = s*40
                rois.append(daisy.Roi(write_offset, write_shape))

            self.overwrite_sections = rois

        extra_config = {
            'voxel_size': self.net_voxel_size,
            'predict_num_core': self.num_cores_per_worker,
            'config_file': config_file,
            'meta_file': meta_file,
        }

        if self.predict_file is not None:
            predict_script = self.predict_file
            if not Path(predict_script).exists():
                # prepend script folder
                prepend = Path(__file__).parent.resolve()
                predict_script = f'{prepend}/{predict_script}'
        else:
            # use the one included in folder
            predict_script = '%s/predict.py' % (self.train_dir)

        assert Path(predict_script).exists(), \
            f'ERROR: predict_script at {predict_script} does not exist!'

        self.sbatch_mem = int(self.num_cores_per_worker*self.mem_per_core)
        if self.batch_num_cores is None:
            self.batch_num_cores = self.num_cores_per_worker

        check_function = \
            lambda b: base_task.check_block(
                b, self.affs_ds, completion_db=self.database,
                recording_block_done=self.recording_block_done,
                logger=logger,
                overwrite_sections=self.overwrite_sections,
                check_datastore=False)

        if self.overwrite:
            check_function = None

        self._write_config(cmd=f"python {predict_script}",
                           extra_config=extra_config)

        return self._prepare_task(
            total_roi=sched_roi,
            read_roi=block_read_roi,
            write_roi=block_write_roi,
            fit='overhang',
            read_write_conflict=False,
            check_fn=check_function,
            upstream_tasks=upstream_tasks,
            max_retries=3,
            )

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        tasks['PredictTask'] = PredictTask(configs['PredictTask']).prepare()
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = PredictTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
