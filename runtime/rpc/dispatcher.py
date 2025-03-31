import copy,json,math,logging,os,queue,re,signal,subprocess,sys,threading,time,traceback
from multiprocessing.pool import ThreadPool

import utils

CUDA_MPS_PIPE_DIRECTORY = "/tmp/nvidia-mps"
CUDA_MPS_LOG_DIRECTORY = "/tmp/nvidia-log"
MAX_CPUS_PER_GPU = 8
LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
ITERATOR_LOG_FORMAT = "[{asctime}] [{event}] [{status}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class Dispatcher:
    def __init__(self,round_duration,
        gpu_ids,
        worker_rpc_client,
        sched_addr,
        sched_port,
        static_run_dir,
        accordion_run_dir,
        gns_run_dir,
        data_dir,
        checkpoint_dir,
        logger,
        use_mps=False,):
        self._logger = logger
        self._thread_pool = ThreadPool()
        self._round_duration = round_duration
        self._worker_rpc_client = worker_rpc_client
        self._sched_addr = sched_addr
        self._sched_port = sched_port
        # self._sched_port = 50080
        self._static_run_dir = static_run_dir
        self._accordion_run_dir = accordion_run_dir
        self._gns_run_dir = gns_run_dir
        self._data_dir = data_dir
        self._checkpoint_dir = checkpoint_dir
        self._gpu_ids = gpu_ids
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)
        self._job_assignments = {}
        self._commands = {} 
        self._lock = threading.Lock()
        #
        self._use_mps = use_mps
        self._mps_initially_enabled = True
    
    def dispatch_jobs(self,jobs,workerid,roundid):
        self._thread_pool.apply_async(self._dispatch_jobs_helper, (jobs, workerid, roundid,))
    
    def _dispatch_jobs_helper(self, jobs, worker_id, round_id):
        job_ids = [job.job_id for job in jobs]
        # self._logger.debug(f'在调度器第 {round_id}轮训练集 {job_ids}需要GPUs {worker_id}')
        gpu_id = self._gpu_queue.get()
        
        self._logger.debug(
            f'在第 {round_id}轮,在服务器 {worker_id}上使用GPU {gpu_id}给作业 {job_ids}进行训练。')
        with self._lock:
            for job_id in job_ids:
                if job_id not in self._job_assignments:
                    self._job_assignments[job_id] = []
                self._job_assignments[job_id].append(gpu_id)
        self._logger.debug(f"基于服务器 {worker_id}开始重塑命令.... ")
        success = True
        commands = []
        for job in jobs:
            try:
                prefix, command = self._construct_command(job, gpu_id, worker_id)
                commands.append((prefix, command))
            except Exception as e:
                self._logger.error(
                    f'由于出现错误 {str(e)},重塑命令失败')
                traceback.print_exc()
                success = False
                break
        if success:
            results = []
            for job, (prefix, command) in zip(jobs, commands):
                self._logger.info(
                    f'在服务器 {worker_id}上运行作业 {job_id}, 调度器轮次为 {round_id},在服务器上的第 {gpu_id}号GPU,'
                    f'作业部分参数如下:prefix: {prefix},mode: {job.mode},MPS thread percentage: {job.mps_thread_percentage}%,\n'
                    f'作业计算command: "{command}')
                results.append(
                    self._thread_pool.apply_async(
                        self.launch_job,
                        (
                            job,
                            prefix,
                            command,
                            worker_id,
                            round_id,
                            gpu_id,
                            job.mode,),))  
            job_descriptions = [result.get() for result in results]
        else:
            job_descriptions = [[job.job_id, 0, 0, ""] for job in jobs]    
        self._gpu_queue.put(gpu_id)
        jobids = [job.job_id for job in jobs]
        self._logger.info(f'作业 {jobids}已经完成,通知调度器')
        self._worker_rpc_client.notify_scheduler(worker_id, job_descriptions)              
    
    def _construct_command(self,job,gpu_id,worker_id):
        checkpoint_dir = os.path.join(
            self._checkpoint_dir, "job_id=%d" % (job.job_id)
        )
        with self._lock:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
        if job.needs_data_dir:
            command = job.command % (self._data_dir)
        else:
            command = job.command
        command = "%s --local_rank %d" % (command, gpu_id)
        command = "%s %s %d" % (command, job.num_steps_arg, job.total_steps)
        command = "%s --checkpoint_dir %s" % (command, checkpoint_dir)
        command = "%s --enable_gavel_iterator" % (command)
        prefix = None
        return prefix,command

    def shutdown(self,shut_down_mps=True):
        self._logger.debug("开始停止 dispatcher 分发器")
        self._kill_jobs()
        self._thread_pool.terminate()
        if self._use_mps and shut_down_mps and not self._mps_initially_enabled:
            self._shutdown_mps()
        self._logger.debug("成功停止 dispatcher 分发器")
    
    def reset(self):
        self._logger.debug("Resetting dispatcher...")
        self._kill_jobs()
        self._job_assignments = {}
        self._thread_pool = ThreadPool()
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)
        self._logger.debug("Finished resetting dispatcher")
    
    def _kill_jobs(self,job_id=None):
        with self._lock:
            if job_id is not None:
                self._logger.debug(f"开始停止作业其序号为 {job_id}...")
            else:
                self._logger.debug(f"成功停止所有作业!")
            if job_id is not None:
                if (
                    job_id not in self._commands
                    or len(self._commands[job_id]) == 0
                ):
                    self._logger.warning(f"作业 {job_id}未找到可以执行的命令")
                    return
                round_id = min(self._commands[job_id].keys())
                pids = []
                for command in self._commands[job_id][round_id]:
                    pid = utils.get_pid_for_job(command)
                    if pid is not None:
                        pids.append(pid)
                self._logger.debug(
                    f'作业 {job_id} 在第 {round_id}的进程标识符为 {round_id}')
                for pid in pids:
                    self._kill_job(pid)
            else:
                for job_id in self._job_assignments:
                    if job_id not in self._commands:
                        continue
                    pids = []
                    for round_id in sorted(self._commands[job_id].keys()):
                        for command in self._commands[job_id][round_id]:
                            pid = utils.get_pid_for_job(command)
                            if pid is not None:
                                pids.append(pid)
                        self._logger.debug(
                            f'作业 {job_id} 在第 {round_id}的进程标识符为 {round_id}')
                        for pid in pids:
                            self._kill_job(pid)
            self._logger.info("成功停止所有 job(s)")
    def _kill_job(self,pid):
        self._logger.info("Killing process {0}".format(pid))
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError as e:
            self._logger.debug(f"无法找到进程号 {pid}")
        except Exception as e:
            self._logger.error(f"由于发生错误{e}无法结束进程 {pid}")

    def _get_job_logger_and_fh(self,job_id,worker_id,round_id):
        checkpoint_dir = os.path.join(self._checkpoint_dir, "job_id=%d" % (job_id))
        # groups_dir = os.path.join(checkpoint_dir,'groups')
        round_dir = os.path.join(checkpoint_dir, "round={0}".format(round_id))
        log_file = os.path.join(round_dir, "worker={0}.log".format(worker_id))
        
        with self._lock:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            # if not os.path.isdir(groups_dir):
            #     os.mkdir(groups_dir)
            if not os.path.isdir(round_dir):
                os.mkdir(round_dir)
        job_logger = logging.getLogger(
            "job={0}_worker={1}_round={2}".format(job_id, worker_id, round_id))        
        job_logger.propagate = False
        job_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                ITERATOR_LOG_FORMAT, datefmt=DATE_FORMAT, style="{"))
        fh.setLevel(logging.DEBUG)
        job_logger.addHandler(fh)
        return job_logger, fh
    
    def _get_iterator_log(self,job_id, worker_id, round_id):
        checkpoint_dir = os.path.join(
            self._checkpoint_dir, "job_id=%d" % (job_id))
        # gavel_dir = os.path.join(checkpoint_dir, "groups")
        round_dir = os.path.join(checkpoint_dir, "round={0}".format(round_id))
        log_file = os.path.join(round_dir, "worker={0}.log".format(worker_id))

        if not os.path.exists(log_file):
            self._logger.error(f'调度器在第 {round_id}轮,作业 {job_id}在服务器 {worker_id}上无法读取logfile')
            return ""
        with open(log_file, "r") as f:
            return f.read().strip()
    
    def _get_steps_and_execution_time(self,job_id, worker_id, round_id):
        checkpoint_dir = os.path.join(
            self._checkpoint_dir, "job_id=%d" % (job_id))
        # gavel_dir = os.path.join(checkpoint_dir, "groups")
        round_dir = os.path.join(checkpoint_dir, "round={0}".format(round_id))
        log_file = os.path.join(round_dir, "worker={0}.log".format(worker_id))
        steps = 0
        execution_time = 0
        with open(log_file, "r") as f:
            for line in f:
                match = re.match("\[(.*)\] \[(.*)\] \[(.*)\]\ ?(.*)", line)
                if match is None:
                    self._logger.error(
                        "Malformed Gavel log file: {0}".format(line))
                    continue
                timestamp = match.group(1)
                event = match.group(2)
                status = match.group(3)
                message = match.group(4)
                if event == "PROGRESS":
                    if status == "STEPS":
                        steps = int(message)
                    elif status == "DURATION":
                        execution_time = float(message)
        return steps, execution_time
        
    
    def launch_job(self, job, prefix, command, worker_id, round_id, gpu_id, mode):
        output = ""
        if mode == "static":
            run_dir = self._static_run_dir
        elif mode == "accordion":
            run_dir = self._accordion_run_dir
        elif mode == "gns":
            run_dir = self._gns_run_dir
        cwd = os.path.join(run_dir, job.working_directory)
        # self._logger.info(
            # f'在第 {round_id}轮,作业 {job.job_id}在服务器 {worker_id}上的数据集地址为 {cwd}')
        job_logger, fh = self._get_job_logger_and_fh(
            job.job_id, worker_id, round_id
        )
        job_logger.info("", extra={"event": "DISPATCHER", "status": "LAUNCH"})    
        with self._lock:
            if job.job_id not in self._commands:
                self._commands[job.job_id] = {}
            if round_id not in self._commands[job.job_id]:
                self._commands[job.job_id][round_id] = set()
            self._commands[job.job_id][round_id].add(command)
        full_command = command
        job_succeeded = False
        try:
            env = copy.deepcopy(os.environ)
            env["GAVEL_JOB_ID"] = str(job.job_id)
            env["GAVEL_WORKER_ID"] = str(worker_id)
            env["GAVEL_ROUND_ID"] = str(round_id)
            env["GAVEL_SCHED_ADDR"] = self._sched_addr
            env["GAVEL_SCHED_PORT"] = str(self._sched_port)
            self._logger.info(f'该计算任务的整个运行命令为 {full_command},根目录地址为 {cwd}')
            proc = subprocess.run(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                shell=True,)
            output = proc.stdout.decode("utf-8").strip()
            # output = proc.stdout.decode('GBK').strip()

            self._logger.info(f'subprocess执行结果为{str(output)}')
            job_succeeded = True
        except subprocess.CalledProcessError as e:
            self._logger.error(f'调度器第 {round_id}轮,作业 {job.job_id}在服务器 {worker_id}上发生CalledProcessError错误')
            traceback.print_exc()
            if e.stdout is not None:
                self._logger.info(f'由于CalledProcessError作业 {job.job_id}发生标准输出 {e.stdout}......')
            if e.stderr is not None:
                self._logger.info(f'由于CalledProcessError作业 {job.job_id}发生错误输出 {e.stderr}......')
        except Exception as e:
            self._logger.error(f'调度器第 {round_id}轮,作业 {job.job_id}在服务器 {worker_id}上发生Exception错误')
            traceback.print_exc()
            self._kill_jobs(job_id=job.job_id)    
        job_logger.info(
            "", extra={"event": "DISPATCHER", "status": "COMPLETE"})    
        job_logger.removeHandler(fh)
        fh.close()
        iterator_log = self._get_iterator_log(job.job_id, worker_id, round_id)
        with self._lock:
            self._commands[job.job_id][round_id].remove(command)
            if len(self._commands[job.job_id][round_id]) == 0:
                del self._commands[job.job_id][round_id]
            if len(self._commands[job.job_id]) == 0:
                del self._commands[job.job_id]
        if not job_succeeded:
            return [job.job_id, 0, 0, iterator_log]
        try:
            (
                completed_steps,
                execution_time,
            ) = self._get_steps_and_execution_time(
                job.job_id, worker_id, round_id
            )
        except Exception as e:
            traceback.print_exc()
            self._logger.error(f'调度器第 {round_id}轮,作业 {job.job_id}在服务器 {worker_id}上计算执行时间和迭代次数失败')
            self._logger.debug(f'调度器第 {round_id}轮,作业 {job.job_id}在服务器 {worker_id}上计算失败后输出为 {output}')
            return [job.job_id, 0, 0, iterator_log]
        return [job.job_id, execution_time, completed_steps, iterator_log]
        