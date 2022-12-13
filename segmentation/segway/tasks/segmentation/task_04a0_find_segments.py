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
from task_03_agglomerate import AgglomerateTask
from lut import LookupTable

logger = logging.getLogger(__name__)


def is_divisible(big_blocksize, small_blocksize):
    assert len(big_blocksize) == len(small_blocksize)
    for a, b in zip(big_blocksize, small_blocksize):
        if a % b != 0:
            return False
    return True


class FindSegmentsGetLocalLUTsTask(BatchTask):

    fragments_file = Parameter()
    fragments_dataset = Parameter()
    merge_function = Parameter()
    edges_collection = Parameter()
    thresholds = Parameter()
    num_workers = Parameter()
    lut_dir = Parameter()

    filedb_file = Parameter(None)
    filedb_dataset = Parameter("filedb")
    filedb_nodes_roi_offset = Parameter(None)
    filedb_edges_roi_offset = Parameter(None)
    filedb_nodes_chunk_size = Parameter()
    filedb_edges_chunk_size = Parameter()

    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    block_size = Parameter()
    indexing_block_size = Parameter(None)

    ignore_degenerates = Parameter(False)

    def prepare(self, upstream_tasks=None):

        # check parameters
        assert is_divisible(
            self.block_size, self.filedb_nodes_chunk_size), \
            "self.filedb_nodes_chunk_size needs to be divisible by blocksize" \
            " for MDSeg to work"
        # assert is_divisible(
        #     self.block_size, self.filedb_edges_chunk_size), \
        #     "self.filedb_edges_chunk_size needs to be divisible by blocksize" \
        #     " for MDSeg to work"

        # sanitize some parameters
        if self.filedb_file is None:
            self.filedb_file = self.fragments_file
        self.last_threshold = self.thresholds[-1]

        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        # sanitize ROIs
        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))
            # if self.filedb_nodes_roi_offset is None:
            #     self.filedb_nodes_roi_offset = (0, 0, 0)
        else:
            total_roi = fragments.roi
            if self.filedb_nodes_roi_offset is None:
                self.filedb_nodes_roi_offset = fragments.roi.get_begin()
        assert fragments.roi.contains(total_roi), (
                "fragments.roi %s does not contain total_roi %s" % (
                    fragments.roi, total_roi))

        if self.filedb_nodes_roi_offset is None:
            # set offset based on affs ROI so that filedb blocks are aligned to
            # boundaries between workers.
            roi_node_offset = [l % m for l, m in zip(self.sub_roi_offset, self.filedb_nodes_chunk_size)]
            roi_edge_offset = [l % m for l, m in zip(self.sub_roi_offset, self.filedb_edges_chunk_size)]
            # TODO: make sure that offsets are consistent between runs
            self.filedb_nodes_roi_offset = tuple(roi_node_offset)
            self.filedb_edges_roi_offset = tuple(roi_edge_offset)

        if self.filedb_edges_roi_offset is None:
            self.filedb_edges_roi_offset = self.filedb_nodes_roi_offset

        read_roi = daisy.Roi((0,)*total_roi.dims, self.block_size)
        write_roi = read_roi

        self.out_dir = os.path.join(
            self.fragments_file,
            self.lut_dir)

        # lut dataset to check completion
        lut_dataset = f'edges_local2frags_{self.merge_function}' \
                      f'_{int(self.last_threshold*100)}'
        self.lut_db = LookupTable(filepath=self.out_dir,
                                  dataset=lut_dataset)

        for threshold in self.thresholds:
            os.makedirs(os.path.join(self.out_dir, "edges_local2local_%s_%d" %
                        (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "edges_local2frags_%s_%d" %
                        (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "nodes_%s_%d" %
                        (self.merge_function, int(threshold*100))), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "seg_frags2local_%s_%d" %
                        (self.merge_function, int(threshold*100))), exist_ok=True)

        extra_config = {
        }

        worker_script = "worker_04a0_find_segments.py"
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
        return self.lut_db.check(block)

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['FindSegmentsGetLocalLUTsTask'].get('no_check_dependency', False):
            tasks = AgglomerateTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['AgglomerateTask'])

        tasks['FindSegmentsGetLocalLUTsTask'] = FindSegmentsGetLocalLUTsTask(
                                        configs['FindSegmentsGetLocalLUTsTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = FindSegmentsGetLocalLUTsTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
