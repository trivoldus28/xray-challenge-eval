import logging
import sys
import os
import os.path as path
import numpy as np
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate, Roi

import base_task
from base_task import Parameter
from batch_task import BatchTask
from task_05a0_find_segments import FindSegmentsGetGlobalLUTsTask
from lut import LookupTable
from util import enumerate_blocks_in_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FindSegmentsGetChunkedGlobalLUTsTask(BatchTask):

    fragments_file = Parameter()
    fragments_dataset = Parameter()
    merge_function = Parameter()
    thresholds = Parameter()
    num_workers = Parameter()
    lut_dir = Parameter()

    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    block_size = Parameter()
    chunk_size = Parameter([2, 2, 2])

    def prepare(self, upstream_tasks=None):

        self.block_size = Coordinate(self.block_size) * Coordinate(self.chunk_size)

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
        else:
            total_roi = fragments.roi
        self.total_roi = total_roi

        assert fragments.roi.contains(total_roi)

        read_roi = daisy.Roi((0,)*total_roi.dims, self.block_size)
        write_roi = read_roi

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        os.makedirs(self.out_dir, exist_ok=True)

        # lut dataset to check completion
        self.last_threshold = self.thresholds[-1]
        lut_dataset = f'seg_local2global_{self.merge_function}' \
                      f'_{int(self.last_threshold*100)}'
        self.lut_db = LookupTable(filepath=self.out_dir,
                                  dataset=lut_dataset)

        # for threshold in self.thresholds:
        #     os.makedirs(os.path.join(
        #             self.out_dir,
        #             "seg_local2global_%s_%d" % (self.merge_function, int(threshold*100)),
        #             ),
        #         exist_ok=True)

        extra_config = {
            # 'db_host': self.db_host,
            # 'db_name': self.db_name,
            # 'fragments_file': self.fragments_file,
            # 'lut_dir': self.lut_dir,
            # 'merge_function': self.merge_function,
            # 'thresholds': self.thresholds,
            # 'chunk_size': self.chunk_size,
            # 'block_size': self.block_size,
            'total_roi_offset': total_roi.get_offset(),
            'total_roi_shape': total_roi.get_shape(),
        }
        worker_script = "worker_05a1_find_segments.py"
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
        # get the last chunk in the chunk
        blocks = enumerate_blocks_in_chunks(
            chunk, self.block_size, self.chunk_size, self.total_roi)
        block = blocks[-1]
        return self.lut_db.check(block)

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['FindSegmentsGetChunkedGlobalLUTsTask'].get('no_check_dependency', False):
            tasks = FindSegmentsGetGlobalLUTsTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['FindSegmentsGetGlobalLUTsTask'])

        tasks['FindSegmentsGetChunkedGlobalLUTsTask'] = FindSegmentsGetChunkedGlobalLUTsTask(
                                        configs['FindSegmentsGetChunkedGlobalLUTsTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks

    # def block_done(self, chunk):

    #     # get the last chunk in the chunk
    #     blocks = enumerate_blocks_in_chunks(
    #         chunk, self.block_size, self.chunk_size, self.total_roi)
    #     block = blocks[-1]

    #     block_id = block.block_id
    #     lookup = 'seg_local2global_%s_%d/%d' % (
    #         self.merge_function,
    #         int(self.last_threshold*100),
    #         block_id
    #         )
    #     out_file = os.path.join(self.out_dir, lookup) + '.npz'
    #     exists = path.exists(out_file)
    #     if not exists:
    #         logger.info("%s not found" % out_file)
    #     return exists


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = FindSegmentsGetChunkedGlobalLUTsTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
