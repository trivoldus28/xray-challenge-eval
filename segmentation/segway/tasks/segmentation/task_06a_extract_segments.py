import logging
import sys
import numpy as np
import pathlib
from pathlib import Path

import daisy
from daisy import Coordinate, Roi

import base_task
from base_task import Parameter
from batch_task import BatchTask
from task_05a1_find_segments import FindSegmentsGetChunkedGlobalLUTsTask
from lut import LookupTable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractSegmentationTask(BatchTask):
    '''Segment .
    These seeds are assumed to belong to the same segment.
    '''

    fragments_file = Parameter()
    fragments_dataset = Parameter()
    out_dataset = Parameter()
    out_file = Parameter()
    context = Parameter([0, 0, 0])
    db_host = Parameter()
    db_name = Parameter()
    thresholds = Parameter()
    merge_function = Parameter()
    lut_dir = Parameter()

    sub_roi_offset = Parameter(None)
    sub_roi_shape = Parameter(None)

    block_size = Parameter()
    write_size = Parameter()

    use_global_roi = Parameter(True)

    def prepare(self, upstream_tasks=None):

        logging.info("Reading fragments from %s", self.fragments_file)
        self.fragments = daisy.open_ds(self.fragments_file,
                                       self.fragments_dataset,
                                       mode='r')

        if self.sub_roi_offset is not None and self.sub_roi_shape is not None:
            total_roi = daisy.Roi(
                tuple(self.sub_roi_offset), tuple(self.sub_roi_shape))

        else:
            total_roi = self.fragments.roi

        write_roi = daisy.Roi((0,)*self.fragments.roi.dims, self.block_size)
        read_roi = write_roi

        delete_ds = False
        if self.overwrite:
            delete_ds = True

        dataset_roi = self.fragments.roi
        if not self.use_global_roi:
            dataset_roi = total_roi

        for threshold in self.thresholds:
            ds = self.out_dataset + "_%.3f" % threshold
            self.segment_ds = daisy.prepare_ds(
                self.out_file,
                ds,
                dataset_roi,
                self.fragments.voxel_size,
                self.fragments.data.dtype,
                write_size=daisy.Coordinate(tuple(self.write_size)),
                force_exact_write_size=True,
                compressor={'id': 'blosc', 'clevel': 3},
                delete=delete_ds,
                )

        last_threshold = self.thresholds[-1]

        extra_config = {
            # 'fragments_file': self.fragments_file,
            # 'fragments_dataset': self.fragments_dataset,
            # 'lut_dir': self.lut_dir,
            # 'merge_function': self.merge_function,
            # 'thresholds': self.thresholds,
            # 'out_dataset': self.out_dataset,
            # 'out_file': self.out_file,
        }
        worker_script = "worker_06a_extract_segments.py"
        if not Path(worker_script).exists():
            # prepend script folder
            prepend = Path(__file__).parent.resolve()
            worker_script = f'{prepend}/{worker_script}'

        self._write_config(cmd=f"python {worker_script}",
                           extra_config=extra_config)

        check_function = self.check_block
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

    def check_block(self, block):

        logger.debug("Checking if block %s is complete..." % block.write_roi)

        if self.segment_ds.roi.intersect(block.write_roi).empty():
            logger.debug("Block outside of output ROI")
            return True

        if self.database.check(block.block_id):
            logger.debug("Skipping block with db check")
            return True

        center_coord = (block.write_roi.get_begin() +
                        block.write_roi.get_end()) / 2
        center_values = self.segment_ds[center_coord]
        s = np.sum(center_values)

        logger.debug("Sum of center values in %s is %f" % (block.write_roi, s))

        done = s != 0
        if done:
            self.recording_block_done(block)

        # TODO: this should be filtered by post check and not pre check
        # if (s == 0):
        #     self.log_error_block(block)

        return done

    @staticmethod
    def make_task(configs, tasks=None):
        '''Instantiate Daisy tasks for the current task and any dependencies'''
        if tasks is None:
            tasks = {}
        upstream_tasks = []

        if not configs['ExtractSegmentationTask'].get('no_check_dependency', False):
            tasks = FindSegmentsGetChunkedGlobalLUTsTask.make_task(configs, tasks)
            upstream_tasks.append(tasks['FindSegmentsGetChunkedGlobalLUTsTask'])

        tasks['ExtractSegmentationTask'] = ExtractSegmentationTask(
                                        configs['ExtractSegmentationTask']).prepare(
                                        upstream_tasks=upstream_tasks)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    configs = base_task.parseConfigs(sys.argv[1:])

    tasks = ExtractSegmentationTask.make_task(configs)
    tasks = [v for k, v in tasks.items()]

    done = daisy.run_blockwise(tasks)
    if done:
        logger.info("Ran all blocks successfully!")
    else:
        logger.info("Did not run all blocks successfully...")
