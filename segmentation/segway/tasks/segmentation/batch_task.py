import os
import datetime
import pymongo

import daisy

from base_task import Parameter, BaseTask


class BatchTask(BaseTask):

    batch_num_cores = Parameter(1)
    batch_time = Parameter("1:00:00")
    batch_mem = Parameter(4)
    batch_partition = Parameter(None)
    batch_account = Parameter(None)
    batch_gpu_type = Parameter(None)
    batch_log_dir = Parameter(None)

    def _write_config(self, cmd, extra_config=None):
        super()._write_config(cmd, extra_config)

    def slurmSetup(
            self, config, actor_script,
            python_module=False,
            python_interpreter='python',
            completion_db_name_extra=None,
            **kwargs):
        '''Write config file and batch file for the actor, and generate
        `new_actor_cmd`. We also keep track of new jobs so to kill them
        when the task is finished.'''

        if not python_module:
            logname = (actor_script.split('.'))[-2].split('/')[-1]
        else:
            logname = (actor_script.split('.'))[-1]

        for prepend in [
                '.',
                # '/n/groups/htem/Segmentation/shared-nondev',
                os.path.dirname(os.path.realpath(__file__))
                ]:
            if os.path.exists(os.path.join(prepend, actor_script)):
                actor_script = os.path.realpath(os.path.join(prepend, actor_script))
                # actor_script = actor_script.replace('/mnt/orchestra_nfs', '/n/groups/htem')
                break
        else:
            logname = (actor_script.split('.'))[-1]

        f = "%s/%s.error_blocks.%s" % (self.log_dir, logname, str(datetime.datetime.now()).replace(' ', '_'))
        self.error_log = open(f, "w")
        self.precheck_log = None

        db_client = pymongo.MongoClient(self.db_host)
        db = db_client[self.db_name]

        if self.completion_db_class_name:
            class_name = self.completion_db_class_name
        else:
            class_name = self.__class__.__name__
        completion_db_name = class_name + '_fb'
        if completion_db_name_extra:
            completion_db_name = completion_db_name + completion_db_name_extra

        if completion_db_name not in db.list_collection_names():
            self.completion_db = db[completion_db_name]
            self.completion_db.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
        else:
            self.completion_db = db[completion_db_name]

        config.update({
            'db_host': self.db_host,
            'db_name': self.db_name,
            'completion_db_name': completion_db_name,
            })

        self.batch_submit_cmd, self.new_actor_cmd = generateActorbatch(
            config,
            actor_script,
            python_module=python_module,
            python_interpreter=python_interpreter,
            log_dir=self.log_dir,
            logname=logname,
            batch_num_cores=self.batch_num_cores,
            batch_time=self.batch_time,
            batch_mem=self.batch_mem,
            batch_gpu_type=self.batch_gpu_type,
            batch_partition=self.batch_partition,
            batch_account=self.batch_account,
            **kwargs)

        # self.logname = logname

    def _init_callback_fn(self, context):
        '''Daisy calls this function on starting a task'''
        super()._init_callback_fn(context)
        # print(f"Submit command: DAISY_CONTEXT={context.to_env()} {self.batch_submit_cmd}")
        print(f"Submit command: DAISY_CONTEXT=TO_IMPLEMENT")


def generateActorbatch(
        config, actor_script,
        python_module,
        log_dir, logname,
        python_interpreter,
        **kwargs):

    config_str = ''.join(['%s' % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))
    try:
        os.makedirs('.run_configs')
    except Exception:
        pass
    config_file = os.path.join(
        '.run_configs', '%s_%d.config' % (logname, config_hash))
    with open(config_file, 'w') as f:
        json.dump(config, f)

    if not python_module:
        run_cmd = ' '.join([
            python_interpreter,
            '%s' % actor_script,
            '%s' % config_file,
            ])
    else:
        run_cmd = ' '.join([
            python_interpreter,
            '-m',
            '%s' % actor_script,
            '%s' % config_file,
            ])

    batch_script = os.path.join('.run_configs', '%s_%d.sh'%(logname, config_hash))
    generatebatchScript(
        batch_script, run_cmd, log_dir, logname,
        **kwargs)

    new_worker_cmd = [
        'sh',
        '%s' % batch_script
        ]

    return run_cmd, new_worker_cmd



def generatebatchScript(
        batch_script,
        run_cmd,
        log_dir,
        logname,
        batch_time="1:00:00",
        batch_num_cores=1,
        batch_mem=6,
        batch_gpu_type=None,
        batch_partition=None,
        batch_account=None,
        ):
    text = []
    text.append("#!/bin/bash")
    text.append("#SBATCH -t %s" % batch_time)

    if batch_gpu_type is not None:
        if batch_partition is None:
            batch_partition = 'gpu'
        if batch_gpu_type == '' or batch_gpu_type == 'any':
            text.append("#SBATCH --gres=gpu:1")
        else:
            text.append("#SBATCH --gres=gpu:{}:1".format(batch_gpu_type))

    if batch_partition is None:
        batch_partition = 'short'
    text.append("#SBATCH -p %s" % batch_partition)

    if batch_account:
        text.append("#SBATCH --account %s" % batch_account)

    text.append("#SBATCH -c %d" % batch_num_cores)
    text.append("#SBATCH --mem=%dGB" % batch_mem)
    text.append("#SBATCH -o {}/{}_%j.out".format(log_dir, logname))
    text.append("#SBATCH -e {}/{}_%j.err".format(log_dir, logname))
    # text.append("#SBATCH -o .logs_batch/{}_%j.out".format(logname))
    # text.append("#SBATCH -e .logs_batch/{}_%j.err".format(logname))

    text.append("")
    # text.append("$*")
    text.append(run_cmd)

    logger.info("Writing batch script %s" % batch_script)
    with open(batch_script, 'w') as f:
        f.write('\n'.join(text))
