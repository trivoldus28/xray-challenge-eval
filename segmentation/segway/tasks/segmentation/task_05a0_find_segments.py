import logging
import sys
import os
import os.path as path
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate, Roi

import base_task
from base_task import Parameter
from batch_task import BatchTask
from lut import LookupTable
from task_04a1_find_segments import FindSegmentsGetLocalEdgesTask

logger = logging.getLogger(__name__)


class FindSegmentsGetGlobalLUTsTask(BatchTask):

    fragments_file = Parameter()
    fragments_dataset = Parameter()
    merge_function = Parameter()
    thresholds = Parameter()
    num_workers = Parameter()
    lut_dir = Parameter()

    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    block_size = Parameter()

    def prepare(self, upstream_tasks=None):

        self.block_size = tuple(self.block_size)

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))

        else:
            fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)
            total_roi = fragments.roi

        assert fragments.roi.contains(total_roi)

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        # lut dataset to check completion
        self.last_threshold = self.thresholds[-1]
        # lut_dataset = f'seg_local2global_{self.merge_function}' \
        #               f'_{int(self.last_threshold*100)}'
        # self.lut_db = LookupTable(filepath=self.out_dir,
        #                           dataset=lut_dataset)

        os.makedirs(self.out_dir, exist_ok=True)

        extra_config = {
            # 'db_host': self.db_host,
            # 'db_name': self.db_name,
            # 'fragments_file': self.fragments_file,
            # 'lut_dir': self.lut_dir,
            # 'merge_function': self.merge_function,
            # 'thresholds': self.thresholds,
        }
        worker_script = "worker_05a0_find_segments.py"
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
            read_roi=total_roi,
            write_roi=total_roi,
            fit='shrink',
            read_write_conflict=False,
            check_fn=check_function,
            upstream_tasks=upstream_tasks,
            max_retries=self.max_retries,
            )

    def block_done(self, block):

        block_id = block.block_id
        lookup = 'seg_local2global_%s_%d_single' % (
            self.merge_function, int(self.last_threshold*100))
        out_file = os.path.join(self.out_dir, lookup) + '.npz'
        logger.debug("Checking %s" % out_file)
        exists = path.exists(out_file)
        return exists

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['FindSegmentsGetGlobalLUTsTask'].get('no_check_dependency', False):
            tasks = FindSegmentsGetLocalEdgesTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['FindSegmentsGetLocalEdgesTask'])

        tasks['FindSegmentsGetGlobalLUTsTask'] = FindSegmentsGetGlobalLUTsTask(
                                        configs['FindSegmentsGetGlobalLUTsTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = FindSegmentsGetGlobalLUTsTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
