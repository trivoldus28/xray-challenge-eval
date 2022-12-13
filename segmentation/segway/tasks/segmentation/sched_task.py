
from base_task import Parameter, BaseTask


class SlurmTask(BaseTask):

    sched_num_cores = daisy.Parameter(1)
    sched_time = daisy.Parameter("1:00:00")
    sched_mem = daisy.Parameter(4)
    sched_partition = daisy.Parameter(None)
    sched_account = daisy.Parameter(None)
    sched_gpu_type = daisy.Parameter(None)
    sched_log_dir = daisy.Parameter(None)

    def slurmSetup(
            self, config, actor_script,
            python_module=False,
            python_interpreter='python',
            completion_db_name_extra=None,
            **kwargs):
        '''Write config file and sbatch file for the actor, and generate
        `new_actor_cmd`. We also keep track of new jobs so to kill them
        when the task is finished.'''

        if not python_module:
            logname = (actor_script.split('.'))[-2].split('/')[-1]
        else:
            logname = (actor_script.split('.'))[-1]

        for prepend in [
                '.',
                '/n/groups/htem/Segmentation/shared-nondev',
                os.path.dirname(os.path.realpath(__file__))
                ]:
            if os.path.exists(os.path.join(prepend, actor_script)):
                actor_script = os.path.realpath(os.path.join(prepend, actor_script))
                actor_script = actor_script.replace('/mnt/orchestra_nfs', '/n/groups/htem')
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

        self.slurmtask_run_cmd, self.new_actor_cmd = generateActorSbatch(
            config,
            actor_script,
            python_module=python_module,
            python_interpreter=python_interpreter,
            log_dir=self.log_dir,
            logname=logname,
            sbatch_num_cores=self.sbatch_num_cores,
            sbatch_time=self.sbatch_time,
            sbatch_mem=self.sbatch_mem,
            sbatch_gpu_type=self.sbatch_gpu_type,
            sbatch_partition=self.sbatch_partition,
            sbatch_account=self.sbatch_account,
            **kwargs)
        self.started_jobs = multiprocessing.Manager().list()
        self.started_jobs_local = []

        self.logname = logname
        self.task_done = False
        self.launch_process_cmd = multiprocessing.Manager().dict()

        self.shared_precheck_blocks = multiprocessing.Manager().list()
        self.shared_error_blocks = multiprocessing.Manager().list()

    def init_callback_fn(self, context):
        '''Daisy calls this function on starting a task'''

        run_cmd = f'{context.to_env()} {' '.join(self.new_worker_cmd)}'
        run_cmd = "cd %s" % os.getcwd() + "; " + run_cmd

        logger.info("Submit command: DAISY_CONTEXT={} {}".format(
                context_str,
                ' '.join(self.new_worker_cmd)))


def generateActorSbatch(
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

    sbatch_script = os.path.join('.run_configs', '%s_%d.sh'%(logname, config_hash))
    generateSbatchScript(
        sbatch_script, run_cmd, log_dir, logname,
        **kwargs)

    new_worker_cmd = [
        'sh',
        '%s' % sbatch_script
        ]

    return run_cmd, new_worker_cmd



def generateSbatchScript(
        sbatch_script,
        run_cmd,
        log_dir,
        logname,
        sbatch_time="1:00:00",
        sbatch_num_cores=1,
        sbatch_mem=6,
        sbatch_gpu_type=None,
        sbatch_partition=None,
        sbatch_account=None,
        ):
    text = []
    text.append("#!/bin/bash")
    text.append("#SBATCH -t %s" % sbatch_time)

    if sbatch_gpu_type is not None:
        if sbatch_partition is None:
            sbatch_partition = 'gpu'
        if sbatch_gpu_type == '' or sbatch_gpu_type == 'any':
            text.append("#SBATCH --gres=gpu:1")
        else:
            text.append("#SBATCH --gres=gpu:{}:1".format(sbatch_gpu_type))

    if sbatch_partition is None:
        sbatch_partition = 'short'
    text.append("#SBATCH -p %s" % sbatch_partition)

    if sbatch_account:
        text.append("#SBATCH --account %s" % sbatch_account)

    text.append("#SBATCH -c %d" % sbatch_num_cores)
    text.append("#SBATCH --mem=%dGB" % sbatch_mem)
    text.append("#SBATCH -o {}/{}_%j.out".format(log_dir, logname))
    text.append("#SBATCH -e {}/{}_%j.err".format(log_dir, logname))
    # text.append("#SBATCH -o .logs_sbatch/{}_%j.out".format(logname))
    # text.append("#SBATCH -e .logs_sbatch/{}_%j.err".format(logname))

    text.append("")
    # text.append("$*")
    text.append(run_cmd)

    logger.info("Writing sbatch script %s" % sbatch_script)
    with open(sbatch_script, 'w') as f:
        f.write('\n'.join(text))


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
