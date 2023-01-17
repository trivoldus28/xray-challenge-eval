import copy
import json
import logging
import multiprocessing
import hashlib
import subprocess
import os
import collections
import pymongo
import numpy as np
from io import StringIO
from jsmin import jsmin

import daisy
import ast

from aggregate_configs import aggregateConfigs
# from segway2.tasks.segmentation.aggregate_configs import aggregateConfigs

logger = logging.getLogger(__name__)


UNDEFINED_DAISY_PARAMETER = object()


class Parameter():

    def __init__(self, default=UNDEFINED_DAISY_PARAMETER):
        self.val = default
        self.user_spec = False

    def set(self, val):
        self.val = val
        self.user_spec = True


class Database():

    def __init__(self, db_host, db_id,
                 table_name="completion_db_col",
                 overwrite=False):

        self.table_name = table_name

        if db_host is None:
            # Use SQLite
            import sqlite3
            self.use_sql = True
            os.makedirs("daisy_db", exist_ok=True)
            self.con = sqlite3.connect(f'daisy_db/{db_id}.db', check_same_thread=False)
            self.cur = self.con.cursor()

            if overwrite:
                self.cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                self.con.commit()

            # check if table exists
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [k[0] for k in self.cur.fetchall()]
            if self.table_name not in tables:
                self.cur.execute(f"CREATE TABLE {self.table_name} (block_id text)")
                self.con.commit()

        else:
            # Use MongoDB
            self.use_sql = False
            self.client = pymongo.MongoClient(db_host)

            if overwrite:
                self.client.drop_database(db_id)

            db = self.client[db_id]
            if self.table_name not in db.list_collection_names():
                self.completion_db = db[self.table_name]
                self.completion_db.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
            else:
                self.completion_db = db[self.table_name]

    def check(self, block_id):

        if self.use_sql:
            block_id = '_'.join([str(s) for s in block_id])
            res = self.cur.execute(
                f"SELECT * FROM {self.table_name} where block_id = '{block_id}'").fetchall()
            if len(res):
                return True
        else:
            if self.completion_db.count_documents({'block_id': block_id}) >= 1:
                return True

        return False

    def add_finished(self, block_id):

        if self.use_sql:
            block_id = '_'.join([str(s) for s in block_id])
            self.cur.execute(f"INSERT INTO {self.table_name} VALUES ('{block_id}')")
            self.con.commit()
        else:
            document = {
                'block_id': block_id
            }
            self.completion_db.insert_one(document)


class BaseTask():

    # common parameters for tasks
    max_retries = Parameter(2)
    no_run_workers = Parameter(False)
    overwrite = Parameter(False)
    no_check_dependency = Parameter(False)
    no_precheck = Parameter(False)
    num_workers = Parameter(1)
    timeout = Parameter(None)
    log_dir = Parameter()

    # db params
    db_host = Parameter()
    db_name = Parameter()
    # completion_db_class_name = Parameter(None)
    config_hash = Parameter(None)

    task_uses_gpus = Parameter(False)

    def __init__(self, config=None, task_id=None):

        if task_id:
            self.task_id = task_id
        else:
            # default task ID is the class name
            self.task_id = type(self).__name__

        self.__init_parameters(config)

    def __init_parameters(self, config):

        self.__daisy_params__ = {}
        self.__inherit_params(self.__class__)

        # apply given config
        for key in config:
            if key in self.__daisy_params__:
                self.__daisy_params__[key].set(config[key])
            else:
                raise RuntimeError(
                        "Key %s is not in the Parameter list for Task %s" %
                        (key, self.task_id))

        # finalize parameters
        for param in self.__daisy_params__:
            val = self.__daisy_params__[param].val
            if val is UNDEFINED_DAISY_PARAMETER:
                raise RuntimeError(
                    f"Parameter `{param}` of task {self.task_id} is unassigned" \
                    " and has no default value.")
            setattr(self, param, val)

        self.task_name = str(self.__class__.__name__)

        # self.__init_config = copy.deepcopy(config)

        if self.config_hash is None:
            config_str = ''.join(['%s' % (v,) for k, v in config.items()
                                 if k not in ['overwrite', 'num_workers',
                                              'no_run_workers']])
            self.config_hash = str(hashlib.md5(config_str.encode()).hexdigest())

        self.config_hash_short = self.config_hash[0:8]

        self.db_table_id = self.task_id

        self.write_config_called = False

    def __inherit_params(self, current_class):
        '''Recursively query and inherit `Parameter`s from base `Task`s.
        Parameters are copied to self.__daisy_params__ to be processed
        downstream.
        If duplicated, `Parameter`s of derived `Task`s will override ones
        from base `Task`s, even if it clears any default. This lets
        inherited `Task` to unset defaults and force user to input new
        values.'''
        for b in current_class.__bases__:
            self.__inherit_params(b)

        for param in current_class.__dict__:
            if isinstance(current_class.__dict__[param], Parameter):
                if (current_class.__dict__[param].val is not
                        UNDEFINED_DAISY_PARAMETER):
                    self.__daisy_params__[param] = copy.deepcopy(
                                                current_class.__dict__[param])
                else:
                    self.__daisy_params__[param] = (
                                                current_class.__dict__[param])

    def _write_config(self, cmd, extra_config=None):
        '''Make a config file for workers. Workers can then be run on the
        command line on potentially a different machine and use this file
        to initialize its variables.

        Args:
            extra_config (``dict``, optional):
                Any extra configs that should be written for workers
        '''

        config = {}
        for param in self.__daisy_params__:
            # config[param] = self.__daisy_params__[param].val
            config[param] = getattr(self, param)

        if extra_config:
            for k in extra_config:
                config[k] = extra_config[k]

        self.config_file = f'.run_configs/' \
                           f'{self.task_id}_{self.config_hash_short}.config'

        self.new_worker_cmd = f'{cmd} {self.config_file}'

        config['completion_db_name'] = self.db_table_id

        os.makedirs('.run_configs', exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

        self.write_config_called = True

    def _prepare_task(
            self,
            total_roi,
            read_roi,
            write_roi,
            check_fn=None,
            fit='shrink',
            read_write_conflict=False,
            upstream_tasks=None,
            max_retries=1,
            ):

        assert self.write_config_called, (
            "`BatchTask._write_config()` was not called")

        print("Processing total_roi %s with read_roi %s and write_roi %s" % (
              total_roi, read_roi, write_roi))

        print("db_host: ", self.db_host)
        print("db_name: ", self.db_name)
        print("db_table_id: ", self.db_table_id)

        if self.overwrite:
            print("Dropping table %s" % self.db_table_id)

            if self.overwrite == 2:
                i = "Yes"
            else:
                i = input("Sure? Yes/[No] ")

            if i == "Yes":
                print("Dropped %s!" % self.db_table_id)
            else:
                print("Aborted")
                exit(0)

        self.database = Database(self.db_host, self.db_name,
                                 table_name=self.db_table_id,
                                 overwrite=self.overwrite)

        return daisy.Task(
            task_id=self.task_id,
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=self._new_worker,
            read_write_conflict=read_write_conflict,
            fit=fit,
            num_workers=self.num_workers,
            max_retries=max_retries,
            check_function=check_fn,
            init_callback_fn=self._init_callback_fn,
            upstream_tasks=upstream_tasks,
        )

    def _new_worker(self):
        '''Run "shell" command to start a new worker'''

        if self.no_run_workers is False:

            assert 'DAISY_CONTEXT' in os.environ, (
                "DAISY_CONTEXT must be defined as an environment variable")

            context_str = os.environ['DAISY_CONTEXT']

            cmd = f'DAISY_CONTEXT={context_str} {self.new_worker_cmd}'

            if self.task_uses_gpus:
                worker_id = int(daisy.Context.from_env().worker_id)
                cuda_dev = os.environ.get('CUDA_VISIBLE_DEVICES', worker_id)
                cmd = f"CUDA_VISIBLE_DEVICES={cuda_dev}" + cmd

            # worker_id = daisy.Context.from_env().worker_id
            # logout = open("%s/%s.%d.out" % (
            #                         self.log_dir, self.logname, worker_id),
            #               'a')
            # logerr = open("%s/%s.%d.err" % (
            #                         self.log_dir, self.logname, worker_id),
            #               'a')
            cp = subprocess.run(cmd,
                                # stdout=logout,
                                # stderr=logerr,
                                shell=True
                                )

    def _init_callback_fn(self, context):
        '''Daisy calls this function on starting a task'''
        print(f"Terminal command: DAISY_CONTEXT={context.to_env()} {self.new_worker_cmd}")

    def recording_block_done(self, block):
        self.database.add_finished(block.block_id)


def parseConfigs(args, aggregate_configs=True):
    global_configs = {}
    # cmdline_configs = {}
    hierarchy_configs = collections.defaultdict(dict)

    default_configs = loadJsonFile(args[0]).get("DefaultConfigs", [])

    for default_config_file in default_configs:
        global_configs = mergeDicts(loadJsonFile(default_config_file), global_configs)

    for config in args:

        if "=" in config:
            key, val = config.split('=')
            if "." in val:
                try: val = float(val)
                except: pass
            else:
                try: val = int(val)
                except: pass
            if '.' in key:
                task, param = key.split('.')
                hierarchy_configs[task][param] = val
            # else:
            #     cmdline_configs[key] = ast.literal_eval(val)

        else:
            new_configs = loadJsonFile(config)
            global_configs = mergeDicts(new_configs, global_configs)
            # keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
            # for k in keys:
            #     if k in global_configs:
            #         if k in new_configs:
            #             global_configs[k].update(new_configs[k])
            #     else:
            #         global_configs[k] = new_configs[k]

            if 'Input' in new_configs and 'config_filename' not in global_configs['Input']:
                global_configs['Input']['config_filename'] = config

    # update global confs with hierarchy conf
    for k in hierarchy_configs.keys():
        if k in global_configs:
            global_configs[k].update(hierarchy_configs[k])
        else:
            global_configs[k] = hierarchy_configs[k]

    # # applying command line input parameters
    # for key in cmdline_configs:
    #     if key in self.__daisy_params__:
    #         self.__daisy_params__[key].set(cmdline_configs[key])
    #     else:
    #         raise RuntimeError(
    #                 "Key %s not found in "
    #                 "Parameter list for Task %s" %
    #                 (key, self.task_id))

    if aggregate_configs:
        aggregateConfigs(global_configs)

    # return (user_configs, global_configs)
    return global_configs


def compute_compatible_roi(
        roi_offset, roi_shape,
        sub_roi_offset, sub_roi_shape,
        roi_context,
        net_context,
        source_roi,
        chunk_size,
        center_roi_offset=False,
        shrink_context=False,
        sched_roi_outside_roi_ok=False,
        voxel_size=None,
        ):
    '''Compute compatible input (schedule) ROI and output (dataset ROI)'''

    roi_context = daisy.Coordinate(roi_context)
    net_context = daisy.Coordinate(net_context)

    if roi_offset is not None and roi_shape is not None:

        dataset_roi = daisy.Roi(
            tuple(roi_offset), tuple(roi_shape))

        if center_roi_offset:
            dataset_roi = dataset_roi.shift(-daisy.Coordinate(tuple(roi_shape))/2)
            dataset_roi = dataset_roi.snap_to_grid(voxel_size, mode="grow")

        sched_roi = dataset_roi.grow(net_context, net_context)
        # assert sched_roi.intersect(source_roi) == sched_roi, \
        #     "input_roi (%s) + net_context (%s) = output_roi (%s) has to be within raw ROI %s" \
        #     % (dataset_roi, net_context, sched_roi, source_roi)
        assert dataset_roi.intersect(source_roi) == dataset_roi, \
            "input_roi (%s) + net_context (%s) = output_roi (%s) has to be within raw ROI %s" \
            % (dataset_roi, net_context, dataset_roi, source_roi)

    elif sub_roi_offset is not None and sub_roi_shape is not None:

        dataset_roi = source_roi  # total volume ROI
        sched_roi = daisy.Roi(
            tuple(sub_roi_offset), tuple(sub_roi_shape))
        # assert dataset_roi.contains(sched_roi)

        if center_roi_offset:
            raise RuntimeError("Unimplemented")
        # need align dataset_roi to prediction chunk size

        output_roi_begin = [k for k in dataset_roi.get_begin()]

        print("dataset_roi:", dataset_roi)
        print("sched_roi:", sched_roi)
        print("chunk_size:", chunk_size)
        dataset_roi.set_offset(tuple(output_roi_begin))
        print("dataset_roi:", dataset_roi)

        assert (dataset_roi.get_begin()[0] - sched_roi.get_begin()[0]) % chunk_size[0] == 0
        assert (dataset_roi.get_begin()[1] - sched_roi.get_begin()[1]) % chunk_size[1] == 0
        assert (dataset_roi.get_begin()[2] - sched_roi.get_begin()[2]) % chunk_size[2] == 0

        if not sched_roi_outside_roi_ok:
            assert dataset_roi.contains(sched_roi), "dataset_roi %s does not contain sched_roi %s" % (dataset_roi, sched_roi)

        sched_roi = sched_roi.grow(net_context, net_context)

    else:

        if center_roi_offset:
            raise RuntimeError("Cannot center ROI if not specified")

        assert roi_offset is None
        assert roi_shape is None
        assert sub_roi_offset is None
        assert sub_roi_shape is None
        # if no ROI is given, we need to shrink output ROI
        # to account for the roi_context
        sched_roi = source_roi
        dataset_roi = source_roi

        if roi_context is None:
            roi_context = net_context

        if shrink_context:
            dataset_roi = dataset_roi.grow(-roi_context, -roi_context)
            sched_roi = sched_roi.grow(-roi_context, -roi_context)
            sched_roi = sched_roi.grow(net_context, net_context)
        else:
            # we'd need to increase sched_roi to accomdocate for the context
            sched_roi = sched_roi.grow(net_context, net_context)

    sched_roi = sched_roi.snap_to_grid(voxel_size, mode='grow')
    dataset_roi = dataset_roi.snap_to_grid(voxel_size, mode='grow')
    return sched_roi, dataset_roi


def align(a, b, stride):
    # align a to b such that b - a is multiples of stride
    assert b >= a
    print(a)
    print(b)
    l = b - a
    print(l)
    l = int(l/stride) * stride
    print(l)
    print(b - l)
    return b - l


def check_block(
        block,
        vol_ds,
        completion_db,
        recording_block_done,
        logger,
        overwrite_sections=None,
        overwrite_mask=None,
        check_datastore=True,
        ):

    logger.debug("Checking if block %s is complete..." % block.write_roi)

    write_roi = vol_ds.roi.intersect(block.write_roi)

    if write_roi.empty:
        logger.debug("Block outside of output ROI")
        return True

    if overwrite_sections is not None:
        for roi in overwrite_sections:
            if roi.intersects(block.write_roi):
                logger.debug("Block overlaps overwrite_sections %s" % roi)
            return False

    if overwrite_mask:
        read_roi_mask = overwrite_mask.roi.intersect(block.read_roi)
        if not read_roi_mask.empty:
            try:
                sum = np.sum(overwrite_mask[read_roi_mask].to_ndarray())
                if sum != 0:
                    logger.debug("Block inside overwrite_mask")
                    return False
            except:
                return False

    if completion_db.check(block.block_id):
        return True

    if check_datastore:
        s = 0
        quarter = (write_roi.get_end() - write_roi.get_begin()) / 4

        # check values of center and nearby voxels
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*1])
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*2])
        s += np.sum(vol_ds[write_roi.get_begin() + quarter*3])
        logger.info("Sum of center values in %s is %f" % (write_roi, s))

        done = s != 0
        if done:
            recording_block_done(block)
        return done

    return False


def loadJsonFile(fin):
    with open(fin, 'r') as f:
        config = json.load(StringIO(jsmin(f.read())))
    return config


def mergeDicts(from_dict, to_dict):

    # merge first level
    for k in from_dict:
        if k not in to_dict:
            to_dict[k] = from_dict[k]
        else:
            # overwrite merge second level
            for kk in from_dict[k]:
                to_dict[k][kk] = from_dict[k][kk]

    return to_dict

