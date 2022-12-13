import logging
import numpy as np
import os
import sys
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate, Roi

import base_task
from base_task import Parameter
from batch_task import BatchTask
from task_01_predict import PredictTask

logger = logging.getLogger(__name__)


class ExtractFragmentTask(BatchTask):

    '''
    Parameters:

        affs_file,
        affs_dataset,
        mask_file,
        mask_dataset (``string``):

            Where to find the affinities and mask (optional).

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.

        fragments_in_xy (``bool``):

            Extract fragments section-wise.

        mask_fragments (``bool``):

            Whether to mask fragments for a specified region. Requires that the
            original sample dataset contains a dataset ``volumes/labels/mask``.
    '''

    affs_file = Parameter()
    affs_dataset = Parameter()
    block_size = Parameter()
    context = Parameter()
    filedb_file = Parameter(None)
    filedb_dataset = Parameter("filedb")
    filedb_roi_offset = Parameter(None)
    num_workers = Parameter()

    # sub_roi is used to specify the region of interest while still allocating
    # the entire input raw volume. It is useful when there is a chance that
    # sub_roi will be increased in the future.
    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    mask_fragments = Parameter(default=False)
    mask_file = Parameter(default=None)
    mask_dataset = Parameter(default=None)

    fragments_file = Parameter()
    fragments_dataset = Parameter()
    fragments_in_xy = Parameter()

    raw_file = Parameter(None)
    raw_dataset = Parameter(None)

    epsilon_agglomerate = Parameter(default=0)

    overwrite_mask_f = Parameter(None)
    overwrite_sections = Parameter(None)

    min_seed_distance = Parameter(None)  # default seed size from Jan

    force_exact_write_size = Parameter(False)

    filter_fragments = Parameter(0.3)

    scheduling_chunks = Parameter([1, 1, 1])
    dataset_chunks = Parameter([1, 1, 1])
    database_chunks = Parameter([1, 1, 1])

    # check block completion by counting nodes in DB
    precheck_with_db = Parameter(False)
    precheck_use_affs = Parameter(False)

    # Experimental feature to expand fragments into post filtering empty areas
    # the # specified is the # of expansion iterations using grey dilation
    experimental_expand_fragments = Parameter(0)

    use_global_roi = Parameter(True)

    def prepare(self, upstream_tasks=None):

        logging.info("Reading affs from %s", self.affs_file)
        self.affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        if self.mask_fragments:
            logging.info("Reading mask from %s", self.mask_file)
            self.mask = daisy.open_ds(self.mask_file, self.mask_dataset,
                                      mode='r')
        else:
            self.mask = None

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        if self.context is None:
            self.context = daisy.Coordinate((0,)*self.affs.roi.dims)
        else:
            self.context = daisy.Coordinate(self.context)

        if self.fragments_in_xy:
            # for CB2
            # if we extract fragments in xy, there is no need to have context in Z
            self.context = [n for n in self.context]
            self.context[0] = 0
            self.context = tuple(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            # get ROI of source
            assert self.raw_file is not None and self.raw_dataset is not None
            source = daisy.open_ds(self.raw_file, self.raw_dataset)
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))

            dataset_roi = source.roi
            if not self.use_global_roi:
                dataset_roi = total_roi

            total_roi = total_roi.grow(self.context, self.context)

        else:
            dataset_roi = self.affs.roi
            total_roi = self.affs.roi.grow(self.context, self.context)

            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = dataset_roi.get_begin()

        self.block_size_original = self.block_size
        self.scheduling_chunks = daisy.Coordinate(self.scheduling_chunks)
        self.dataset_chunks = daisy.Coordinate(self.dataset_chunks)

        assert self.database_chunks == [1, 1, 1], "Unsupported for now"
        self.database_chunks = daisy.Coordinate(self.database_chunks)

        dataset_blocksize = daisy.Coordinate(self.block_size)*self.dataset_chunks
        database_blocksize = daisy.Coordinate(self.block_size)*self.database_chunks

        if self.filedb_roi_offset is None:
            # set offset based on affs ROI so that filedb blocks are aligned to
            # boundaries between workers.
            roi_node_offset = [l % m for l, m in zip(self.sub_roi_offset, database_blocksize)]
            # TODO: make sure that offsets are consistent between runs
            self.filedb_roi_offset = tuple(roi_node_offset)

        # prepare fragments dataset
        voxel_size = self.affs.voxel_size
        self.fragments_out = daisy.prepare_ds(
            self.fragments_file,
            self.fragments_dataset,
            dataset_roi,
            voxel_size,
            np.uint64,
            # daisy.Roi((0, 0, 0), self.block_size),
            write_size=tuple(dataset_blocksize),
            force_exact_write_size=self.force_exact_write_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=delete_ds,
            )

        if self.filedb_file is None:
            self.filedb_file = self.fragments_file

        self.rag_provider = daisy.persistence.FileGraphProvider(
            directory=os.path.join(self.filedb_file, self.filedb_dataset),
            chunk_size=database_blocksize,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            save_attributes_as_single_file=True,
            roi_offset=self.filedb_roi_offset,
            nodes_chunk_size=database_blocksize,
            )

        self.overwrite_mask = None
        if self.overwrite_mask_f:
            # force precheck = False for any ROI with any voxel in mask = 1
            self.overwrite_mask = daisy.open_ds(
                self.overwrite_mask_f, "overwrite_mask")

        if self.overwrite_sections is not None:
            write_shape = [k for k in total_roi.get_shape()]
            write_shape[0] = 40
            write_shape = tuple(write_shape)

            rois = []
            for s in self.overwrite_sections:
                write_offset = [k for k in total_roi.get_begin()]
                write_offset[0] = s*40
                rois.append(daisy.Roi(write_offset, write_shape))

            self.overwrite_sections = rois

        # estimate min seed distance base on voxel size
        if self.min_seed_distance is None:
            if voxel_size[2] == 8:
                # for 40x8x8
                self.min_seed_distance = 8
            elif voxel_size[2] == 16:
                # for 40x16x16
                self.min_seed_distance = 5
            elif voxel_size[2] == 50:
                # for 50x50x50
                self.min_seed_distance = 5
            else:
                self.min_seed_distance = 10

        extra_config = {
            'block_size_original': self.block_size_original,
            'database_blocksize': database_blocksize,
        }

        worker_script = "worker_fragment_extract.py"
        if not Path(worker_script).exists():
            # prepend script folder
            prepend = Path(__file__).parent.resolve()
            worker_script = f'{prepend}/{worker_script}'

        self._write_config(cmd=f"python {worker_script}",
                           extra_config=extra_config)

        check_function = self.check
        if self.overwrite:
            check_function = None

        scheduling_block_size = daisy.Coordinate(self.block_size)*self.scheduling_chunks
        read_roi = daisy.Roi((0,)*total_roi.dims,
                             scheduling_block_size).grow(self.context, self.context)
        write_roi = daisy.Roi((0,)*total_roi.dims, scheduling_block_size)

        return self._prepare_task(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            fit='shrink',
            read_write_conflict=False,
            check_fn=check_function,
            upstream_tasks=upstream_tasks,
            max_retries=self.max_retries,
            )

    def check(self, block):

        # if self.overwrite:
        #     return False

        if self.overwrite_sections is not None:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            for roi in self.overwrite_sections:
                if roi.intersects(read_roi_mask):
                    logger.debug("Block overlaps overwrite_sections %s" % roi)
                    return False

        if self.overwrite_mask:
            read_roi_mask = self.overwrite_mask.roi.intersect(block.read_roi)
            if not read_roi_mask.empty():
                try:
                    sum = np.sum(self.overwrite_mask[read_roi_mask].to_ndarray())
                    if sum != 0:
                        logger.debug("Block inside overwrite_mask")
                        return False
                except:
                    return False

        if self.precheck_with_db:
            center = (block.write_roi.get_begin() + block.write_roi.get_end()) / 2
            shrinked_roi = daisy.Roi(center, (1, 1, 1))
            shrinked_roi = shrinked_roi.snap_to_grid(self.block_size_original)

            if self.rag_provider.num_nodes(shrinked_roi):
                # self.recording_block_done(block)
                return True
            elif self.rag_provider.num_nodes(block.write_roi):
                # just making sure and check the entire block
                return True
            else:
                return False

        if self.database.check(block.block_id):
            return True

        return False

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['ExtractFragmentTask'].get('no_check_dependency', False):
            tasks = PredictTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['PredictTask'])

        tasks['ExtractFragmentTask'] = ExtractFragmentTask(
                                        configs['ExtractFragmentTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = ExtractFragmentTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
