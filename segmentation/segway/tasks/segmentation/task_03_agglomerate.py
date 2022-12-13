import logging
import numpy as np
import sys
import os
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate, Roi

import base_task
from base_task import Parameter
from batch_task import BatchTask
from task_02_extract_fragments import ExtractFragmentTask

logger = logging.getLogger(__name__)


class AgglomerateTask(BatchTask):
    '''
    Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

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

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary below.
    '''

    affs_file = Parameter()
    affs_dataset = Parameter()
    fragments_file = Parameter()
    fragments_dataset = Parameter()
    indexing_block_size = Parameter(None)
    block_size = Parameter()
    filedb_nodes_chunk_size = Parameter(None)
    filedb_edges_chunk_size = Parameter(None)
    context = Parameter()
    db_host = Parameter()
    db_name = Parameter()
    filedb_file = Parameter(None)
    filedb_dataset = Parameter("filedb")
    filedb_roi_offset = Parameter(None)
    filedb_edges_roi_offset = Parameter(None)
    num_workers = Parameter()
    merge_function = Parameter()
    threshold = Parameter(default=1.0)
    edges_collection = Parameter()  # debug
    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    def prepare(self, upstream_tasks=None):

        logging.info("Reading affs from %s", self.affs_file)
        affs = daisy.open_ds(self.affs_file, self.affs_dataset, mode='r')

        logging.info("Reading fragments from %s", self.fragments_file)
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset, mode='r')

        assert fragments.data.dtype == np.uint64

        # shape = affs.shape[1:]
        self.context = daisy.Coordinate(self.context)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            total_roi = total_roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*total_roi.dims,
                                 self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*total_roi.dims, self.block_size)
            # if self.filedb_roi_offset is None:
            #     self.filedb_roi_offset = (0, 0, 0)
            # else:
            # self.filedb_roi_offset = tuple([l % m for l, m in zip(affs.roi.get_begin(), )])

        else:
            total_roi = affs.roi.grow(self.context, self.context)
            read_roi = daisy.Roi((0,)*affs.roi.dims, self.block_size).grow(self.context, self.context)
            write_roi = daisy.Roi((0,)*affs.roi.dims, self.block_size)
            if self.filedb_roi_offset is None:
                self.filedb_roi_offset = affs.roi.get_begin()

        # if self.use_sub_roi_offset_for_db:
        #     assert self.sub_roi_offset is not None, (
        #             "if use_sub_roi_offset_for_db is True, sub_roi_offset cannot be None")
        #     self.filedb_roi_offset = self.sub_roi_offset

        if self.filedb_file is None:
            self.filedb_file = self.fragments_file

        if self.filedb_roi_offset is None:
            # set offset based on affs ROI so that filedb blocks are aligned to
            # boundaries between workers.
            roi_node_offset = [l % m for l, m in zip(self.sub_roi_offset, self.filedb_nodes_chunk_size)]
            roi_edge_offset = [l % m for l, m in zip(self.sub_roi_offset, self.filedb_edges_chunk_size)]
            # TODO: make sure that offsets are consistent between runs
            self.filedb_roi_offset = tuple(roi_node_offset)
            self.filedb_edges_roi_offset = tuple(roi_edge_offset)

        if self.filedb_edges_roi_offset is None:
            self.filedb_edges_roi_offset = self.filedb_roi_offset

        # open RAG DB
        self.rag_provider = daisy.persistence.FileGraphProvider(
            directory=os.path.join(self.filedb_file, self.filedb_dataset),
            chunk_size=self.filedb_edges_chunk_size,
            mode='r+',
            directed=False,
            position_attribute=['center_z', 'center_y', 'center_x'],
            save_attributes_as_single_file=True,
            nodes_roi_offset=self.filedb_roi_offset,
            nodes_chunk_size=self.filedb_nodes_chunk_size,
            edges_chunk_size=self.filedb_edges_chunk_size,
            edges_roi_offset=self.filedb_edges_roi_offset,
            )

        extra_config = {
        }

        worker_script = "worker_agglomerate.py"
        if not Path(worker_script).exists():
            # prepend script folder
            prepend = Path(__file__).parent.resolve()
            worker_script = f'{prepend}/{worker_script}'

        self._write_config(cmd=f"python {worker_script}",
                           extra_config=extra_config)

        check_function = self.block_done
        if self.overwrite:
            check_function = None

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

    def block_done(self, block):

        # checking with DB is somehow not reliable
        return self.database.check(block.block_id)

        if not self.database.check(block.block_id):
            return False

        if self.rag_provider.has_edges(block.write_roi):
            return True

        # no nodes found, means an error in fragment extract; skip
        return False

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['AgglomerateTask'].get('no_check_dependency', False):
            tasks = ExtractFragmentTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['ExtractFragmentTask'])

        tasks['AgglomerateTask'] = AgglomerateTask(
                                        configs['AgglomerateTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = AgglomerateTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
