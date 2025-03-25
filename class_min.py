from asyncio import sleep
import collections,copy,faulthandler,os,pickle,scipy,threading,time,datetime,random,math,logging,queue
from typing import OrderedDict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, thread
from copy import deepcopy
from collections import OrderedDict

# 导入本地文件
from job_table import JobTable
import utils
from runtime.rpc import class_client,class_server

LOG_FORMAT = "{name}:{levelname} {message}"
CLASS_PORT = 50080
BASE_JOB_PORT = 60570
MAX_PORT = 65535
INFINITY = int(1e9)
MAX_FAILED_ATTEMPTS = 6
EMA_ALPHA = 0.5

class CLASSmin(object):
    def __init__(self,policy,class_info,jobs,ips,ports,jobsid,clusterinfos,throughputs_file=None,
    seed=0,time_per_iteration=20,profiling_percentage=1.0,
    num_reference_models=len(JobTable),
    max_rounds=None,
    log_level="DEBUG",):
        #时间 
        self._start_timestamp = time.time()
        self._current_timestamp = self._start_timestamp
        #日志
        logger = logging.getLogger(__name__)
        logging_level_dict = {"INFO": logging.INFO, "DEBUG": logging.DEBUG}
        logger.setLevel(logging_level_dict[log_level])
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        logger.addHandler(ch)
        
        logger.addHandler(logging.FileHandler(os.path.join(os.path.join(os.path.dirname(__file__),'logging'),f'classmin_output_{policy}.txt'),mode='w',))
        self._orig_logger = logger
        self._logger = SchedulerAdapter(
                logger,
                {"class": self, "start_timestamp": datetime.datetime.now()},
            )
        self._logging_handler = ch
        loc = "at {addr}:{port}".format(
                addr=utils.get_ip_address(), port=CLASS_PORT)

        self._logger.info(
            "Running scheduler {loc} with the following args: "
            "policy={policy}, seed={seed}, "
            "time_per_iteration={time_per_iteration}, "
            "profiling_percentage={profiling_percentage}, "
            "num_reference_models={num_reference_models}".format(
                loc=loc,
                policy=policy,
                seed=seed,
                time_per_iteration=time_per_iteration,
                profiling_percentage=profiling_percentage,
                num_reference_models=num_reference_models,
            ))
        # 错误监控
        faulthandler.enable()
        path = './logging/'
        assert path is not None
        f = open(path+"class_stack_trace.log", "w")
        faulthandler.dump_traceback_later(30, repeat=True, file=f, exit=False)
        # 进程锁
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        
        
# 定义各类参数
        self._need_to_update_allocation = True
        self._num_completed_rounds = 0
        self._policy = utils.get_policy(policy)
        self._job_id_counter = 0
        self._jobs = jobs
        self._max_rounds = max_rounds
        self._completed_jobs = set()
        self._steps_run_so_far = {}
        self._job_time_so_far = {}
        self._throughputs = {}
        self._job_id_to_job_type = {}
        self._original_bs = {}
        self._original_num_steps = {}
        self._job_types = {}
        self._num_failures_per_job = {}
        self._total_steps_run = {}
        self._cumulative_run_time = {}
        
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
        self._bs_flags = {}
        self._num_scheduled_rounds = OrderedDict()
        self._num_queued_rounds = OrderedDict()
        self._job_start_round = {}
        self._job_end_round = {}
        self._steps_run_in_current_lease = {}
        
        throughputs_file_new = r'.\throughputs\actual_throughput_new.json'
        throughputs_file_old = r'.\throughputs\tacc_throughputs.json'
        self._oracle_throughputs = utils.read_all_throughputs_json_v2(throughputs_file_old)
        self._worker_type_to_worker_id_mapping = {}
        self._worker_id_to_worker_type_mapping = {}
        self._worker_types = set()
        for i in ['k80','p100']:
            self._worker_types.add(i)
        
        for job in self._jobs:
            id = job.job_id
            self._num_queued_rounds[id] = 0
            self._num_scheduled_rounds[id] = 0
            self._steps_run_in_current_lease[id] = 0
            self._steps_run_so_far[id] = 0
            self._total_steps_run[id] = 0
            self._per_job_latest_timestamps[id] = 0
            self._per_job_start_timestamps[id] = self.get_current_timestamp()
            self._throughputs[id] = {}
            for workertype in self._worker_types:
                self._throughputs[id][workertype] = self._oracle_throughputs[workertype][(self._jobs[id].job_type,self._jobs[id].scale_factor)]

        self._cluster_spec = {}
        self.server_mapping_to_nums = {}
        for i in range(len(clusterinfos)):
            if clusterinfos[i].type not in self._cluster_spec:
                self._cluster_spec[clusterinfos[i].type] = []
            self._cluster_spec[clusterinfos[i].type].append(clusterinfos[i].indexs)
            self.server_mapping_to_nums[clusterinfos[i].indexs] = clusterinfos[i].nums
        print(self._cluster_spec,self.server_mapping_to_nums)

        rpc_clients = []
        ipss = []

        ips = ips[0].ips
        ports = ports[0].ids
        for j in range(len(ips)):
            client = class_client.ClassRpcClient(ips[j],ports[j])
            rpc_clients.append(client)
        self._all_rpc_clients = rpc_clients
        self._worker_connections = {}
        # 需要修改
        index = 0
        self._worker_ids = []

        for i,k in enumerate(self._cluster_spec):
            if k not in self._worker_type_to_worker_id_mapping:
                self._worker_type_to_worker_id_mapping[k] = []
            for j in self._cluster_spec[k]:
                self._worker_type_to_worker_id_mapping[k].append(j)
                self._worker_ids.append(j)
                self._worker_connections[j] = self._all_rpc_clients[j]
                self._worker_id_to_worker_type_mapping[j] = k
        
        self._per_round_schedule = []
        
        self._priorities = {}
        self._deficits = {}
        self._worker_time_so_far = {}
        
        self._worker_id_counter = 0
        self._cumulative_worker_time_so_far = {}
        
        self._available_worker_ids = SetQueue()
        
        self._worker_start_times = {}
        self._time_per_iteration = time_per_iteration
        
        self._lease_update_requests = {}
        self._max_steps = {}
        self._completed_jobs_in_current_round = set()
        self._running_jobs = set()
        self._jobs_with_extended_lease = set() 
        self._in_progress_updates = {}
        self._throughput_timeline = {}
        self._job_completion_times = {}
        self._job_type_to_job_ids = {}
        self.new_throughputs_dict = {} 
        callbacks = {
            "InitJob": self._init_job_callback,
            "UpdateLease": self._update_lease_callback,
            "Done": self._done_callback,}
        # 调度器守护进程
        # 1.alloc计算,计算每个小class中的分配情况
        self._allocation_thread = threading.Thread(target=self._allocation_thread)
        self._allocation_thread.daemon = True
        self._allocation_thread.start()
        # 2.服务端启动
        self.server_thread = threading.Thread(target=class_server.server, args=(CLASS_PORT, callbacks))
        self.server_thread.daemon = True
        self.server_thread.start()
        # 3.调度启动
        self._mechanism_thread = threading.Thread(target=self._schedule_with_rounds)
        self._mechanism_thread.daemon = True
        self._mechanism_thread.start()

    # 守护进程函数定义
    def _allocation_thread(self):
        while True:
            self._scheduler_cv.acquire()
            while not self._need_to_update_allocation:
                self._scheduler_cv.wait()
            # 计算每个训练任务的状态
            state = {}
            state['scale_factors'] = {job.job_id: self._jobs[job.job_id].scale_factor for job in self._jobs}
            state["priority_weights"] = {job.job_id: self._jobs[job.job_id].priority_weight for job in self._jobs}
            state["num_steps_remaining"] = {job.job_id: self._get_remaining_steps(job.job_id) - self._steps_run_in_current_lease[job.job_id] for job in self._jobs}
            state["times_since_start"] = {job.job_id: self.get_current_timestamp() - self._per_job_start_timestamps[job.job_id] for job in self._jobs}
            state["throughputs"] = copy.deepcopy(self._throughputs)
            state["per_round_schedule"] = copy.deepcopy(self._per_round_schedule)
            self.gpus_state_nums = {}
            for i in self._cluster_spec:
                if i not in self.gpus_state_nums:
                    self.gpus_state_nums[i] = 0
                for j in self._cluster_spec[i]:
                    nums = self.server_mapping_to_nums[j]
                    self.gpus_state_nums[i] += nums
            state["cluster_spec"] = copy.deepcopy(self.gpus_state_nums)
            self._scheduler_cv.release()
            # 计算任务分配
            self._scheduler_cv.acquire()
            allocation = self._policy.get_allocation(state)
            self._logger.info(f"分配矩阵: {allocation}")
            self._allocation = allocation
            self._need_to_update_allocation = False
            self._allocation_changed_since_last_time_reset = True
            self._scheduler_cv.notify_all()
            self._scheduler_cv.release()
    
    def _schedule_jobs_on_workers_help(self,worker_types):
        already_scheduled_jobs = set()
        scheduled_jobs = {}
        num_workers_left = {}
        for worker_type in worker_types:
            scheduled_jobs[worker_type] = []
            num_workers = self.gpus_state_nums[worker_type]
            num_workers_left[worker_type] = num_workers
        jobqueue = []
        for worktype in worker_types:
            for job in self._jobs:
                jobqueue.append((job.job_id,worktype))
        for (jobid,workertype) in jobqueue:
            if self._policy.name.startswith("FIFO") and self._allocation[jobid][workertype] <=0:
                continue
            if num_workers_left[workertype] < self._jobs[jobid].scale_factor:
                continue
            elif jobid in already_scheduled_jobs:
                continue
            else:
                num_workers_left[workertype] -= self._jobs[jobid].scale_factor
                scheduled_jobs[workertype].append((jobid,self._jobs[jobid].scale_factor))
                already_scheduled_jobs.add(jobid)
        return scheduled_jobs,already_scheduled_jobs
                
    def _schedule_jobs_on_workers(self):
        to_remove = []
        worker_types = ["v100", "p100", "k80"]
        for i, worker_type in enumerate(worker_types):
            if worker_type not in self._worker_type_to_worker_id_mapping:
                to_remove.append(i)
        for i in reversed(to_remove):
            worker_types.pop(i)           
        scheduled_jobs,already_scheduled_jobs = self._schedule_jobs_on_workers_help(worker_types)
        worker_state = {}
        new_worker_assignments = collections.OrderedDict()
        for workertype in worker_types:
            scheduled_jobs[workertype].sort(key=lambda x: x[1], reverse=True)
            worker_ids = copy.deepcopy(
                self._worker_type_to_worker_id_mapping[workertype])
            worker_state[workertype] = {
                "worker_ids": worker_ids,
                "assigned_worker_ids": set(), 
                "server_id_ptr": 0,}
        for i in already_scheduled_jobs:
            new_worker_assignments[i] = []
        for workertype in worker_types:
            for i in range(len(scheduled_jobs[workertype])):
                if len(worker_state[workertype]["worker_ids"]) >= scheduled_jobs[workertype][i][1]:
                    jobid = scheduled_jobs[workertype][i][0]
                    scalefactors  = scheduled_jobs[workertype][i][1]
                    for j in worker_state[workertype]["worker_ids"]:
                        if j not in worker_state[workertype]["assigned_worker_ids"]and scalefactors >0:
                            new_worker_assignments[jobid].append(j)
                            scalefactors -= 1
                            worker_state[workertype]["assigned_worker_ids"].add(j)
        self._logger.info(f'当前的集群机器分配矩阵:{new_worker_assignments}')
        activejobs = [job.job_id for job in self._jobs]
        for jobid in activejobs:
            if jobid in already_scheduled_jobs:
                self._num_scheduled_rounds[jobid] += 1
            else:
                self._num_queued_rounds[jobid] += 1
        return new_worker_assignments
            
    def _schedule_with_rounds(self):
        self._port_offset = 0
        self._logger.info('--'*50)
        self._logger.info('按轮次进行调度')
        self._scheduler_cv.acquire()
        while len(self._jobs) ==0 or len(self._worker_ids) == 0:
            self._scheduler_cv.wait()
            self._scheduler_cv.notifyAll()
        for workerid in self._worker_ids:
            self._available_worker_ids.put(workerid)
        while self._need_to_update_allocation:
            self._scheduler_cv.wait()
        self._current_worker_assignments = self._schedule_jobs_on_workers()
        print('qwert',self._current_worker_assignments)
        self._scheduler_cv.release()
        while True:
            current_round = self._num_completed_rounds
            self._logger.info(f'---------------  开始第 {current_round}轮训练 ------------')
            self._logger.info(f'本轮任务分配情况{self._current_worker_assignments}')
            self._current_round_start_time = self.get_current_timestamp()
            for (job_id, worker_ids) in self._current_worker_assignments.items():
                self._lease_update_requests[job_id] = []
                self._max_steps[job_id] = None
                scale_factor = len(worker_ids)
                master_addr = None
                master_job_ports = []
                if scale_factor > 1:
                    master_addr = self._worker_connections[worker_ids[0]].addr
                    master_job_ports.append(BASE_JOB_PORT + self._port_offset)
                    self._logger.info(f"作业{job_id}端口为{BASE_JOB_PORT + self._port_offset}")
                    self._port_offset += 1
                    self._port_offset %= MAX_PORT - BASE_JOB_PORT
                for i,workerid in  enumerate(worker_ids):
                    self._construct_commands(job_id,master_addr,master_job_ports,scale_factor,i,workerid,current_round)
            time.sleep(self._time_per_iteration+self._current_round_start_time-self.get_current_timestamp())
            self.get_next_round_allocations() 
            self._current_worker_assignments = self._next_worker_assignments
            self._num_completed_rounds += 1
                # time.sleep(self._current_round_start_time + self._time_per_iteration - self.get_current_timestamp())     
    
    def get_next_round_allocations(self):
        self._scheduler_cv.acquire()
        if not self._need_to_update_allocation:
            self._need_to_update_allocation = True 
            self._scheduler_cv.notifyAll()
        self._scheduler_cv.release()
        time.sleep(self._time_per_iteration*0.1)
        self._next_worker_assignments = self._schedule_jobs_on_workers()
        for (jobid,workerid) in self._current_worker_assignments.items():
            if (jobid,workerid) in self._next_worker_assignments.items():
                self._jobs_with_extended_lease.add(jobid)

    def _construct_commands(self,jobid,master_addr,master_job_ports,scale_factor,i,worker_id,current_round):
        job_descriptions = []
        self._logger.info(f"Job {jobid}")
        num_steps = self._jobs[jobid].total_steps
        command = self._jobs[jobid].command
        mps_thread_percentage = self._jobs[jobid].mps_thread_percentage
        if scale_factor > 1:
            command = (
            "%s --master_addr %s ""--master_port %d ""--world_size %d ""--rank %d"
            % (command,master_addr, master_job_ports,scale_factor,i,))
        job_descriptions.append((jobid,command,self._jobs[jobid].working_directory,self._jobs[jobid].needs_data_dir,
        self._jobs[jobid].num_steps_arg,num_steps,self._jobs[jobid].mode,mps_thread_percentage,))
        self._logger.info(
                f"Round {current_round}, running job {jobid} on worker {worker_id}, job_descriptions: {job_descriptions}")
        self._worker_connections[worker_id].run_job(job_descriptions, worker_id, current_round)
        
    def is_done(self,jobs_to_complete):
        jobs_to_complete_set = set()
        for i in jobs_to_complete[0].ids:
            jobs_to_complete_set.add(i)
        with self._scheduler_lock:
            if (self._max_rounds is not None and self._num_completed_rounds >= self._max_rounds):
                return True
            elif jobs_to_complete is not None:
                return jobs_to_complete_set.issubset(self._completed_jobs)

    def _remove_job(self,jobid):
        self._completed_jobs.add(jobid)
        duration = (self._per_job_latest_timestamps[jobid]- self._per_job_start_timestamps[jobid])
        
        self._job_completion_times[jobid] = duration
        self._job_type_to_job_ids[(self._jobs[jobid].job_type,self._jobs[jobid].scale_factor)].remove(jobid)
        del self._steps_run_so_far[jobid]
        del self._job_time_so_far[jobid]
        del self._throughputs[jobid]
        del self._job_id_to_job_type[jobid]
        del self._num_failures_per_job[jobid]
        self._job_end_round[jobid] = self._num_completed_rounds
        if jobid in self._in_progress_updates:
            del self._in_progress_updates[jobid]
        if jobid in self._lease_update_requests:
            del self._lease_update_requests[jobid]
        if jobid in self._max_steps:
            del self._max_steps[jobid]
        if jobid in self._jobs_with_extended_lease:
            self._jobs_with_extended_lease.remove(jobid)
        del self._steps_run_in_current_lease[jobid]
        del self._jobs[jobid]
        self._scheduler_cv.acquire()
        self._need_to_update_allocation = True
        self._scheduler_cv.notifyAll()
        self._scheduler_cv.release()
        self._logger.info(f'作业 {jobid}信息被清除完毕,集群中剩余作业数量 {len(self._jobs)}')

    def _update_throughput(self,jobid,workertype,all_num_steps,all_execition_time):
        if jobid not in self._throughputs:
            return
        if jobid not in self._throughput_timeline.keys():
            self._throughput_timeline[jobid] = OrderedDict()
        current_round = self._num_completed_rounds
        if all_execition_time <= 0:
            new_throughput = 0.0
        else:
            new_throughput = all_num_steps/all_execition_time
        bs_in_current_lease = self._jobs[jobid].batch_size
        self._throughput_timeline[jobid][current_round] = (new_throughput,bs_in_current_lease)
        old_throughput = self._throughputs[jobid][workertype]['null']
        if all_execition_time <=0:
            new_throughput = 0
        else:
            new_throughput = all_num_steps/all_execition_time
        if old_throughput != INFINITY:
            new_throughput *= EMA_ALPHA
            new_throughput += (1-EMA_ALPHA) * old_throughput
        self._throughputs[jobid][workertype] = new_throughput
        self._logger.info(f'任务 {jobid}在 {workertype}上的吞吐量由 {old_throughput}转变为 {new_throughput}')
        if not 0.8 < new_throughput/old_throughput <1.2:
            self._logger.warning(f'任务 {jobid}吞吐量发生了巨大的变化')
            if round(new_throughput/old_throughput,5) == 0.5:
                self._logger.warning(f'计算资源未被利用,任务 {jobid}最新的吞吐量为0')

    def _print_schedule_summary(self):
        completed_jobs = set()
        for job_id, worker_ids in self._current_worker_assignments.items():
            worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
            if (job_id in self._completed_jobs_in_current_round):
                completed_jobs.add(job_id)
            self._logger.debug('作业{0}在{1}个{2}上完成'.format(job_id,len(worker_ids),worker_type))

    def _get_remaining_steps(self, job_id):
        steps_run_so_far = self._total_steps_run[job_id]
        return self._jobs[job_id].total_steps - steps_run_so_far

    def get_current_timestamp(self, in_seconds=False):
        if in_seconds:
            return time.time() - self._start_timestamp
        else:
            return time.time()

# 回调函数callbacks
     
    def _init_job_callback(self,job_id):
        jobid = job_id
        # with self._scheduler_lock:
        self._per_job_latest_timestamps[jobid] = self.get_current_timestamp()
        self._running_jobs.add(jobid)
        remaining_steps = self._get_remaining_steps(jobid)
        remaining_steps = int(math.ceil(remaining_steps/self._jobs[jobid].scale_factor))
        remaining_time_in_current_round = max(self._current_round_start_time+self._time_per_iteration-self.get_current_timestamp(),0)
        self._logger.debug(f'作业 {jobid}:本轮剩余时间 {remaining_time_in_current_round}')
        if remaining_time_in_current_round > 0:
            self._logger.debug(f'初始化作业 {jobid},迭代次数 {remaining_steps}, 本轮剩余时间 {remaining_time_in_current_round}')
            return (remaining_steps,remaining_time_in_current_round,0)
        else:
            ttt = self._time_per_iteration - 3.0
            self._logger.debug(f'初始化作业 {jobid},迭代次数 {remaining_steps}, 本轮剩余时间 {ttt}')
            return (remaining_steps,self._time_per_iteration - 1.0,0)
    
    def _update_lease_callback(self,jobid,workerid,steps,duration,maxsteps,maxduration):
        # with self._scheduler_lock:
        
        run_time_so_far = int(sum(self._cumulative_run_time[jobid].values())/ self._jobs[jobid].scale_factor)
        deadline = int(self._jobs[jobid].duration * 2)
        if jobid not in self._lease_update_requests:
            self._lease_update_requests[jobid] = []
        update_id = len(self._lease_update_requests[jobid])
        self._lease_update_requests[jobid].append((steps,duration,maxsteps,maxduration))
        scale_factor = self._jobs[jobid].scale_factor
        remaining_steps = self._get_remaining_steps(jobid)
        remaining_steps = int(math.ceil(remaining_steps / scale_factor))
        remaining_time_in_current_round = max((self._current_round_start_time + self._time_per_iteration -  self.get_current_timestamp()),0)
        self._logger.info(f'\n')
        self._logger.info(f'作业 {jobid} 开始Update-Lease 已经运行的时间{run_time_so_far},最迟运行时间{deadline}')
        if steps == 0 or duration == 0:
            self._logger.info(f'更新租约 任务 {jobid},剩余训练 {remaining_steps},剩余时间 {remaining_time_in_current_round}')
            return (remaining_steps,remaining_time_in_current_round,run_time_so_far,deadline)
        if jobid in self._jobs_with_extended_lease:
            updated_lease_duration = duration
            # duration是已经工作了的时间
            updated_lease_duration += remaining_time_in_current_round
            updated_lease_duration += self._time_per_iteration
            self._logger.info(f'更新租约 任务 {jobid},剩余训练 {maxsteps},剩余时间 {updated_lease_duration}')
            return (maxsteps,updated_lease_duration,run_time_so_far,deadline)
        
        if scale_factor ==1:
            self._logger.info(f'更新租约 任务 {jobid},剩余训练 {maxsteps},剩余时间 {duration +remaining_time_in_current_round }')
            return (maxsteps,duration+remaining_time_in_current_round,run_time_so_far,deadline)
        else:
            if update_id ==0:
                assert self._max_steps[jobid] is None
                throughput = steps/duration
                self._max_steps[jobid] = min(remaining_steps,steps+int(remaining_time_in_current_round * throughput))
                self._logger.info(f'更新租约 任务{jobid},剩余训练 {self._max_steps[jobid]},剩余时间 {INFINITY}')
                return (self._max_steps[jobid],INFINITY,run_time_so_far,deadline)
            else:
                while True:
                    if self._max_steps[jobid] is not None:
                        break
                    self._logger.info(f'任务{jobid} 等待续约')
                assert self._max_steps[jobid] is not None
                self._logger.info(f'更新租约 任务{jobid},剩余训练 {self._max_steps[jobid]},剩余时间 {INFINITY}')
                return (self._max_steps[jobid],INFINITY,run_time_so_far,deadline)
        
    def _done_callback(self,jobid,workerid,all_num_steps,all_execution_time,all_iterator_logs=None):
        to_remove = []
        # with self._scheduler_lock:
        jobid = jobid[0]
        all_num_steps = all_num_steps[0]
        all_execution_time = all_execution_time[0]
        if all_execution_time == 0:
            throughputs = 0
        else:
            throughputs =  all_num_steps/all_execution_time
        if self._worker_id_to_worker_type_mapping[workerid] not in self.new_throughputs_dict[self._jobs[jobid].job_type]:
            self.new_throughputs_dict[self._jobs[jobid].job_type][self._worker_id_to_worker_type_mapping[workerid]] = throughputs
        else:
            self.new_throughputs_dict[self._jobs[jobid].job_type][self._worker_id_to_worker_type_mapping[workerid]]  +=  throughputs
            self.new_throughputs_dict[self._jobs[jobid].job_type][self._worker_id_to_worker_type_mapping[workerid]] /= 2 
        if int(workerid) not in self._cumulative_run_time[jobid]:
            self._cumulative_run_time[jobid][workerid] = 0
        self._cumulative_run_time[jobid][workerid] += np.max(all_execution_time)
        if jobid in self._jobs.keys():
            run_time_so_far = sum(self._cumulative_run_time[jobid].values()) /self._jobs[jobid].scale_factor
            self._logger.debug(f'任务{jobid} 需要硬件{workerid}，迄今为止运行时间{run_time_so_far}')
            is_over_deadline = run_time_so_far > int(self._jobs[jobid].duration * 1.5)
            if is_over_deadline:
                self._logger.warning(f'任务{jobid}超过最大时间限制')
        self._scheduler_cv.acquire()
        while(jobid not in self._current_worker_assignments or jobid in self._completed_jobs_in_current_round):
            if jobid not in self._current_worker_assignments:
                self._logger.warning(f'作业{jobid}尚未被调度')
                return
            self._logger.warning(f'等待作业完成')
            self._scheduler_cv.wait()
        self._scheduler_cv.notifyAll()
        self._scheduler_cv.release()
        if not jobid in self._jobs:
            self._logger.info(f'作业{jobid}已经完成了')
            return
        workertype = self._worker_id_to_worker_type_mapping[workerid]
        self._logger.debug(f'增加硬件序号为{workerid}')
        self._available_worker_ids.put(workerid)
        scale_factor = len(self._current_worker_assignments[jobid])
        if jobid not in self._in_progress_updates:
            self._in_progress_updates[jobid] = []
        self._in_progress_updates[jobid].append((workerid,all_num_steps,all_execution_time,all_iterator_logs))
        if len(self._in_progress_updates[jobid]) < scale_factor:
            return
        else:
            self._in_progress_updates[jobid].sort(key=lambda x: x[0])
            self._completed_jobs_in_current_round.add(jobid)
            micro_task_succeeded = True
            all_worker_ids = [x[0] for x in self._in_progress_updates[jobid]]
            all_worker_ids.sort()
            for i , update in enumerate(self._in_progress_updates[jobid]):
                all_num_steps_ = update[1]
                all_execution_times_ = update[2]
                all_iterator_logs_ = update[3]
                if jobid not in self._jobs:
                    continue
                elif all_num_steps_ <= 0 and all_execution_times_ <= 0:
                    micro_task_succeeded = False
                    self._logger.debug(f'作业 {jobid}失败,执行时间和次数均小于等于0{all_num_steps_},{all_execution_times_}')
                    continue
                all_num_steps += all_num_steps_
                all_execution_time = max(all_execution_time,all_execution_times_)
        self._in_progress_updates[jobid] = []
        self._lease_update_requests[jobid] = []
        self._max_steps[jobid] = None
        if jobid in self._jobs:
            self._per_job_latest_timestamps[jobid] = self.get_current_timestamp()
        if not micro_task_succeeded:
            self._logger.info(f'[Micro-task failed] 任务{jobid}')
            if jobid in self._jobs:
                self._num_failures_per_job[jobid] += 1
                self._logger.info(f'任务 {jobid} 任务失败，已经失败了{self._num_failures_per_job[jobid]}次')
                if self._num_failures_per_job[jobid] >= MAX_FAILED_ATTEMPTS:
                    start_time = self._per_job_start_timestamps[jobid]
                    finish_time = self._per_job_latest_timestamps[jobid]
                    duration = finish_time - start_time
                    self._logger.info(f'任务失败{jobid},开始时间{start_time},结束时间{finish_time},运行时间{duration}')
                    to_remove.append(jobid)
            self._need_to_update_allocation = True
        else:
            self._num_failures_per_job[jobid] = 0
            if jobid not in self._jobs:
                self._logger.debug('任务 {jobid}不活跃，不需要更新元信息')
            if jobid in self._running_jobs:
                self._steps_run_so_far[jobid][workertype] += all_num_steps
                self._total_steps_run[jobid] += all_num_steps
                self._steps_run_in_current_lease[jobid] = 0
                remaining_steps = self._get_remaining_steps(jobid)
                if remaining_steps <=0 or is_over_deadline:
                    start_time = self._per_job_start_timestamps[jobid]
                    finish_time = self._per_job_latest_timestamps[jobid]
                    duration = finish_time - start_time
                    self._logger.info(f'[Job succeeded] 任务 {jobid},开始时间 {start_time},完成时间 {finish_time},持续时间 {duration}')
                    to_remove.append(jobid)
                else:
                    self._logger.debug(f'任务 {jobid}尚未完成,剩余训练迭代次数 {remaining_steps}')
            max_execution_time = np.max(all_execution_time)
            if jobid in self._job_time_so_far:
                self._job_time_so_far[jobid][workertype] += max_execution_time
                self._worker_time_so_far[workertype] += max_execution_time
            for workerid in all_worker_ids:
                self._cumulative_worker_time_so_far[workerid] += max_execution_time
        
        self._update_throughput(jobid,workertype,all_num_steps,all_execution_time)
        jobids  = [jobid]
        # 动态调整批大小batchsize,未实装
        # for jobid in jobids:
        #     self._scale_bs_and_iters(jobid)
        for jobid in to_remove:
            self._logger.debug(f'任务 {jobid}结束,从调度集群脱离')
            self._remove_job(jobid)
        # if jobid in self._jobs_with_extended_lease:
        #     self._redispatched_worker_assignments
        for jobid in jobids:
            if jobid is None:
                continue
            if (self._bs_flags[jobid]['big_bs'] or self._bs_flags[jobid]['small_bs']):
                self._need_to_update_allocation = True
            self._bs_flags[jobid]['big_bs'] = False
            self._bs_flags[jobid]['small_bs'] = False
        # self._scheduler_cv.notifyAll()
    


class SchedulerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        scheduler = self.extra["class"]
        timestamp = scheduler.get_current_timestamp(in_seconds=True)
        timestamp = self.extra["start_timestamp"] + datetime.timedelta(
                0, timestamp
            )
        return "[%s] %s" % (timestamp, msg), kwargs
    
class SetQueue(queue.Queue):
    def get(self, block=True, timeout=None, item=None):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get(item)
            self.not_full.notify()
            return item
    def get_nowait(self, item=None):
        return self.get(block=False, item=item)

    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):
        self.queue.add(item)

    def _get(self, item):
        if item is None:
            return self.queue.pop()
        elif item in self.queue:
            self.queue.remove(item)
            return item
        else:
            return None

    def __contains__(self, item):
        with self.mutex:
            return item in self.queue
        

def server(policy,classinfo,jobsid,ips,ports,jobs,clusterinfos):
    classmin = CLASSmin(policy,classinfo,jobs,ips,ports,jobsid,clusterinfos)

    while True:
        if classmin.is_done(jobsid):
            break
        else:
            time.sleep(5)
    time.sleep(5)