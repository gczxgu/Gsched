from ast import Return
import collections,copy,faulthandler,os,pickle,scipy,threading,time,datetime,random,math,logging,queue
from tokenize import group
import pathlib
from typing import OrderedDict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, thread
from copy import deepcopy
from collections import OrderedDict
import cvxpy as cp

# 导入本地文件
from job_table import JobTable
import utils
from runtime.rpc import scheduler_client,scheduler_server
# from class_min import CLASSmin

LOG_FORMAT = "{name}:{levelname} {message}"
SCHEDULER_PORT = 50060
BASE_JOB_PORT = 60570
MAX_PORT = 65535
INFINITY = int(1e9)
MAX_FAILED_ATTEMPTS = 6
EMA_ALPHA = 0.5
ALPHA_TYPES = 1.5

dataset_size_dict = {
    "ResNet-18": 50000,  # cifar10
    "ResNet-50": 100000,  # imagenet
    "Transformer": 10000,  # multi30k
    "LM": 59675,  # wikitext2
    "Recommendation": 117907,  # ml-20m
    "CycleGAN": 6287,  # monet2photo
    "A3C": 4,  # no dataset
}

class Scheduler(object):
    def __init__(self,policy,throughputs_file=None,
    seed=0,time_per_iteration=360,profiling_percentage=1.0,
    num_reference_models=len(JobTable),
    per_instance_type_prices_dir=None,
    available_clouds=[],
    assign_SLOs=False,
    enable_global_queue=False,
    expected_num_workers=None,
    minimum_time_between_allocation_resets=1000,
    max_rounds=None,
    pickle_file=None,
    shockwave_config=None,
    log_level="INFO",):
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
        
        logger.addHandler(logging.FileHandler(os.path.join(os.path.join(os.path.dirname(__file__),'logging'),f'console_output_{policy.name}.txt'),mode='w',))
        self._orig_logger = logger
        self._logger = SchedulerAdapter(
                logger,
                {"scheduler": self, "start_timestamp": datetime.datetime.now()},
            )
        self._logging_handler = ch
        loc = "at {addr}:{port}".format(
                addr=utils.get_ip_address(), port=SCHEDULER_PORT)
        self._logger.info(
            "Running scheduler {loc} with the following args: "
            "policy={policy}, seed={seed}, "
            "time_per_iteration={time_per_iteration}, "
            "profiling_percentage={profiling_percentage}, "
            "num_reference_models={num_reference_models}".format(
                loc=loc,
                policy=policy.name,
                seed=seed,
                time_per_iteration=time_per_iteration,
                profiling_percentage=profiling_percentage,
                num_reference_models=num_reference_models,
            ))
        # 错误监控
        faulthandler.enable()
        path = './logging/'
        assert path is not None
        f = open(path+"stack_trace.log", "w")
        faulthandler.dump_traceback_later(30, repeat=True, file=f, exit=False)
        # 进程锁
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        
        
# 定义各类参数
        self._need_to_update_allocation = False
        self._num_completed_rounds = 0
        self._policy = policy
        self._job_id_counter = 0
        self._jobs = {}
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
        self._worker_types = set()
        self._worker_types_list = []
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
        self._bs_flags = {}
        self._num_scheduled_rounds = OrderedDict()
        self._num_queued_rounds = OrderedDict()
        self._job_start_round = {}
        self._job_end_round = {}
        self._steps_run_in_current_lease = {}
        self._per_round_schedule = []
        self._cluster_spec = {}
        self._worker_ids = []
        self._worker_type_to_worker_id_mapping = {}
        self._priorities = {}
        self._deficits = {}
        self._worker_time_so_far = {}
        self._oracle_throughputs,_ = utils.read_throughputs_single(throughputs_file)
        self._worker_id_counter = 0
        self._cumulative_worker_time_so_far = {}
        self._worker_id_to_worker_type_mapping = {}
        self._available_worker_ids = SetQueue()
        self._worker_connections = {}
        self._worker_start_times = {}
        self._time_per_iteration = time_per_iteration
        self._all_rpc_clients = []
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
        self._threshold_workers_dict= {}
        self._threshold_list_first = []
        self._threshold_list_second = []
        self._label_workers_dict = {}
        self._label_list = []
        self.groups_dict_to_jobs = {}
        self.groups_th_array = []
        self.throughputs_ids_mapping_to_jobs_ids = {}
        
        self._available_server_ids = set()
        self._server_type_to_server_id_mapping = {}
        self._server_id_to_server_type_mapping = {}
        self._server_id_counter = 0
        self._server_connections = {}
        self._servers_ids = []
        self._servers_ips = {}
        self._servers_ports = {}
        
        
        
        # 模拟硬件登陆
        self._worker_types.add('k80')
        # self._worker_types.add('p100')

        self._worker_types_list = ['k80','p100']
        self._worker_types_list = ['k80']
        self._threshold_list_first = [1.3,1.0]
        self._threshold_list_second = [1.2,1.0]
        self._server_type_to_server_id_mapping = {'k80':[0,2],'p100':[1,3]}
        self._server_type_to_server_id_mapping = {'k80':[0]}
        self._server_id_to_server_type_mapping = {0:'k80',1:'p100',2:'k80',3:'p100'}
        self._server_id_to_server_type_mapping = {0:'k80'}
        self._servers_ids = [0,1,2,3]    
        # self._available_server_ids.update([0,1,2,3])    
        self._available_server_ids.add(0)
        self._servers_ips = {0:'192.168.93.1',1:'123',2:'123',3:'456'}
        self._servers_ips = {0:'192.168.93.1'}
        self._servers_ports = {0:50071,1:123,2:123,3:456}
        self._servers_ports = {0:50071}
        self.centers = {3: 0,0:0}
        self.server_nums_maapping_gpus = {0:1}
        
        
        callbacks = {
            "RegisterWorker": self._register_worker_callback,}
        # 调度器守护进程
        # 1.alloc计算
        self._allocation_thread = threading.Thread(target=self._allocation_nodes)
        self._allocation_thread.daemon = True
        self._allocation_thread.start()
        # 2.服务端启动
        self.server_thread = threading.Thread(target=scheduler_server.server, args=(SCHEDULER_PORT, callbacks))
        self.server_thread.daemon = True
        self.server_thread.start()
        # 3.class端交流分配
        self._class_thread = threading.Thread(target=self.boot_classmins)
        self._class_thread.daemon = True
        self._class_thread.start()

    def _allocation_nodes(self):
        while True:
            self._scheduler_cv.acquire()
            while not self._need_to_update_allocation:
                self._scheduler_cv.wait()
            state = {}
            self.classsfy()
            state['scale_factors'] = {ids: self._jobs[ids].scale_factor for ids in self._jobs}
            state["throughputs"] = copy.deepcopy(self._throughputs)
            state["cluster_spec"] = copy.deepcopy(self._cluster_spec)
            allocation = self._get_allocation(state)
            self._logger.info(f"一级分配矩阵: {allocation}")
            self._allocation = allocation
            self._allocations_mapping_to_ids()
            self._need_to_update_allocation = False
            self._allocation_changed_since_last_time_reset = True
            self._scheduler_cv.notify_all()
            self._scheduler_cv.release()
    
    def _get_allocation(self,state):
        print(self.groups_th_array)
        x= cp.Variable((np.array(self.groups_th_array).shape),integer=True)
        arraygg = self.groups_gpus
        array111 = self.groups_th_array
        objective = cp.Maximize(cp.sum(cp.multiply(array111,x)-cp.sum_squares(arraygg-cp.sum(x,axis=1)*4)))
        constraints = []
        for k,v in enumerate(self._worker_types_list):
            constraints.append (cp.sum(x[:,k]) == len(self._server_type_to_server_id_mapping[v]))
        for i in range(len(self.groups_gpus)):
            constraints.append(cp.sum(x[i,:]) >=1)
        constraints.append(x>=0)
        prob = cp.Problem(objective,constraints)
        prob.solve(solver= cp.GUROBI)
        self.first_alloc = x.value
        alloc = np.zeros_like(x.value)
        for i in range(len(x.value)):
            for j in range(len(x.value[0])):
                alloc[i][j] = round(x.value[i][j])
        print('第一级分配矩阵',alloc)
        return x.value
    
    def classsfy(self):
        throughput = self._oracle_throughputs
        throughput_array = np.zeros((len(self._jobs),len(self._worker_types)))
        ratios_array = np.zeros((len(self._jobs),len(self._worker_types)))
        avg_ratios_array = np.zeros((1,len(self._worker_types)))
        comps_array = np.zeros((len(self._jobs),len(self._worker_types)))
        throughputs_ids_mapping_to_jobs_ids = {}
        for k,v in enumerate(self._jobs):
            models = self._jobs[v].model
            bs = self._jobs[v].batch_size
            scalefctors = self._jobs[v].scale_factor
            size = dataset_size_dict[models]
            factors = size/bs
            throughputs_ids_mapping_to_jobs_ids[k] = v
            for k2,v2 in enumerate(throughput):
                if v2 in self._worker_types:
                    for v3 in throughput[v2]:
                        r1,r2 = utils.get_re_results(v3)
                        if r1.group(1) == models and int(r1.group(2)) == bs and int(r2.group(2)) == scalefctors:
                            throughput_array[k][k2] = throughput[v2][v3]/factors*100
        # 模仿加速比计算公式
        for k,v in enumerate(throughput_array):
            throughput_array[k] = v/min(v)
        # 模仿效率计算公式
        for k,v in enumerate(throughput_array):
            sums = sum(v)
            try:
                for j in range(len(v)):
                    ratios_array[k][j] = v[j] / sums
            except Exception as e:
                self._logger.error(f'由于出现错误{str(e)}该作业无法在集群下进行')
        
        avg_ratios_array = np.mean(ratios_array,axis=0)
        comps_array = ratios_array/avg_ratios_array
            
        
        groups_nums = math.ceil(len(self._worker_types) * ALPHA_TYPES)
        groups_dict_to_jobs = {}
        groups_th_array = np.zeros((3*groups_nums,len(self._worker_types)))
        
        str1 = 'groups'
        for i in range(3*groups_nums):
            key = f'{str1+str(i)}'
            groups_dict_to_jobs[key] = []
        
        # 这里体现决策树的分类依据,相关数值后续进行修改. 
        for k,v in enumerate(ratios_array):
            index_max = np.argmax(v)
            v = np.delete(v, index_max)
            if len(v) == 0:
                key = f'{str1+str(index_max*3)}'
                groups_dict_to_jobs[key].append(k)
                continue
            index_max_2 = np.argmax(v)
            if comps_array[k][index_max] >= self._threshold_list_first[index_max] or index_max in np.where(self._label_list == np.min(self._label_list))[0]:
                key = f'{str1+str(index_max*3)}'
                groups_dict_to_jobs[key].append(k)
            elif comps_array[k][index_max_2] >= self._threshold_list_second[index_max] or index_max_2 in np.where(self._label_list == np.min(self._label_list))[0]:
                key = f'{str1+str(index_max*3+1)}'
                groups_dict_to_jobs[key].append(k)
            else:
                key = f'{str1+str(index_max*3+2)}'
                groups_dict_to_jobs[key].append(k)
        for k,v in enumerate(groups_th_array):
            key = f'{str1+str(k)}'
            for j in groups_dict_to_jobs[key]:
                v += throughput_array[j]
            if len(groups_dict_to_jobs[key]) >0:
                v /= len(groups_dict_to_jobs[key])
                groups_th_array[k] = v
        for v in groups_th_array:
            if sum(v) != 0:
                continue
            
        groups_th_array = [row for row in groups_th_array if sum(row) !=0]
        none_zero_groups_dict_to_jobs = {key:value for key,value in groups_dict_to_jobs.items() if len(value) != 0}
        id_none_zero_groups_dict_to_jobs = {}
                
        self.groups_dict_to_jobs = none_zero_groups_dict_to_jobs
        for k,v in enumerate(groups_dict_to_jobs):
            id_none_zero_groups_dict_to_jobs[k] = groups_dict_to_jobs[v]
        self.id_groups_dict_to_jobs = id_none_zero_groups_dict_to_jobs
        # {3: [0, 1, 2, 3, 4]}
        self.id_groups_dict_to_jobs = {key:value for key,value in id_none_zero_groups_dict_to_jobs.items() if len(value) != 0}
        
        self.groups_th_array = groups_th_array
                
        groups_gpus = np.zeros((len(self.groups_dict_to_jobs),))
        for k,v in enumerate(self.groups_dict_to_jobs):
            for l in self.groups_dict_to_jobs[v]:
                ids = throughputs_ids_mapping_to_jobs_ids[l]
                groups_gpus[k] += self._jobs[ids].scale_factor
        self.groups_gpus = groups_gpus
        self.throughputs_ids_mapping_to_jobs_ids = throughputs_ids_mapping_to_jobs_ids
        
        
    def add_job(self,job):
        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            jobid = self._job_id_counter
            job._jobid = jobid
            self._job_id_counter += 1
            self._jobs[jobid] = job
            self._steps_run_so_far[jobid] = {}
            self._job_time_so_far[jobid] = {}
            self._throughputs[jobid] = {}
            self._job_id_to_job_type[jobid] = (job.job_type,job.scale_factor)
            if (job.job_type,job.scale_factor) not in self._job_type_to_job_ids:
                self._job_type_to_job_ids[(job.job_type,job.scale_factor)]=set()
            self._job_type_to_job_ids[(job.job_type,job.scale_factor)].add(jobid)
            self._original_bs[jobid] = job.batch_size
            self._original_num_steps[jobid] = job.total_steps
            self._job_types[jobid] = job.job_type
            self._num_failures_per_job[jobid] = 0
            self._total_steps_run[jobid] = 0
            self._cumulative_run_time[jobid] = {}
            self.new_throughputs_dict[job.job_type] = {}
            for workertype in self._worker_types:
                self._job_time_so_far[jobid][workertype] = (0)
                self._steps_run_so_far[jobid][workertype] = 0
                self._throughputs[jobid][workertype] = self._oracle_throughputs[workertype][(job.job_type,job.scale_factor)]
            self._per_job_start_timestamps[jobid] = current_timestamp
            self._per_job_latest_timestamps[jobid] = None
            self._need_to_update_allocation = True
            self._bs_flags[jobid] = {"big_bs": False, "small_bs": False}
            self._num_scheduled_rounds[jobid] = 0
            self._num_queued_rounds[jobid] = 0
            self._job_start_round[jobid] = self._num_completed_rounds
            self._steps_run_in_current_lease[jobid] = 0
            self._per_job_start_timestamps[jobid] = self.get_current_timestamp()
            self._logger.info('新作业记录如下：作业序号{0},\n'
            '开始时间：{1},完成迭代次数：{2},作业类型:{3}\n'.format(jobid,self._per_job_start_timestamps[jobid], self._total_steps_run[jobid],self._job_id_to_job_type[jobid]))
            self._scheduler_cv.notifyAll()

    def is_done(self,jobs_to_complete):
        with self._scheduler_lock:
            if (self._max_rounds is not None and self._num_completed_rounds >= self._max_rounds):
                return True
            elif jobs_to_complete is not None:
                return jobs_to_complete.issubset(self._completed_jobs)

    def get_current_timestamp(self, in_seconds=False):
        if in_seconds:
            return time.time() - self._start_timestamp
        else:
            return time.time()

    def _allocations_mapping_to_ids(self):
        allocations = self._allocation
        idgroups = self.id_groups_dict_to_jobs
        limit_array = np.zeros((len(self._servers_ids),len(self._servers_ids)))
        available_server_ids = copy.deepcopy(self._available_server_ids)
        # 关注集群中位置、通信等影响
        # 暂时没有建模
        allocations_ids = {}
        for k,v in enumerate(idgroups):
            if k not in allocations_ids:
                allocations_ids[v] = []
            for k2,v2 in enumerate(allocations[k]):
                if v2 ==0 :
                    continue
                else:
                    counters = 0
                    ids = self._server_type_to_server_id_mapping[self._worker_types_list[k2]]
                    for j in ids:
                        if j in available_server_ids:
                            allocations_ids[v].append(j)
                            counters += 1
                            available_server_ids.remove(j)
                        if counters == v2:
                            break
        allocations_types = {}
        for k,v in enumerate(idgroups):
            if k not in allocations_types:
                allocations_types[v] = {}
                for type in self._worker_types:
                    allocations_types[v][type] = []
            ids  = allocations_ids[v]
            for id in ids:
                type = self._server_id_to_server_type_mapping[id]
                allocations_types[v][type].append(id)
        
        self._allocation_types_ids = allocations_types
        self._allocatione_ids = allocations_ids
    
    def commu_time(self):
        self.commu_time_array = np.zeros((len(self._servers_ids),len(self._servers_ids)))
        centers = []
        self.centers = {}
        for v in self.id_groups_dict_to_jobs:
            sums = []
            for j in self.id_groups_dict_to_jobs[v]:
                sums.append(sum(self.commu_time_array[j]))
            index = np.argmin(sums)
            centers.append(index)
        for k,v in self._allocatione_ids:
            self.centers[v] = centers[k]
            
      
    def boot_classmins(self):
        self._logger.info('启动所有的class_mins')
        self._scheduler_cv.acquire()

        while len(self._jobs) == 0 or len(self._servers_ids) ==0:
            self._scheduler_cv.wait()
            self._scheduler_cv.notify_all()
        for serverid in self._servers_ids:
            self._available_server_ids.add(serverid)
        while self._need_to_update_allocation:
            self._scheduler_cv.wait()
        #  {3: [0, 2, 1, 3]}
        groups_allocation_ids = self._allocatione_ids
        self._scheduler_cv.release()
        # while True:
        self._logger.info('组调度器集群重新进行分组分配')
        self._logger.info(f'本次组分配情况:{groups_allocation_ids}')
        for (groupid,serverids) in groups_allocation_ids.items():
            intids = []
            for i in serverids:
                intids.append(int(i))
            policy = 'fifo'
            classinfo = self._allocation_types_ids[groupid]
            jobs = self.id_groups_dict_to_jobs[groupid]
            job_descriptions = []
            cluster_infos = []
            for jobid in jobs:
                job_descriptions.append((jobid,self._jobs[jobid].job_type,self._jobs[jobid].command,self._jobs[jobid].working_directory,self._jobs[jobid].needs_data_dir,\
                    self._jobs[jobid].num_steps_arg,self._jobs[jobid].total_steps,self._jobs[jobid].mode,self._jobs[jobid].mps_thread_percentage,\
                    self._jobs[jobid].scale_factor))
            for i in classinfo:
                for j in classinfo[i]:
                    num = self.server_nums_maapping_gpus[j]
                    cluster_infos.append((i,j,num))
            # rpc_clients = []
            # for k in serverids:
            #     rpc_clients.append(self._all_rpc_clients[k])
            ips,ports = [],[]
            for k in serverids:
                ips.append(self._servers_ips[k])
                ports.append(self._servers_ports[k])
            class_ip = self._servers_ips[self.centers[groupid]]
            class_port = self._servers_ports[self.centers[groupid]]
            rpc_client = scheduler_client.SchedulerRpcClient(class_ip,class_port)
            rpc_client.run_class(policy,serverids,jobs,ips,ports,job_descriptions,cluster_infos,class_ip,class_port)

                

# 回调函数callbacks
    def _register_worker_callback(self,workertype,numgpus=1,ip_addr=None,port=None):
        self._logger.info('登记硬件资源类型为{0},局域网内地址:{1}:{2}'.format(workertype,ip_addr,port))
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
        self._all_rpc_clients.append(rpc_client)
        server_counter = self._server_id_counter
        self._servers_ips[server_counter] = ip_addr
        self._servers_ports[server_counter] = port
        self._servers_ids.append(server_counter)
        self._available_server_ids.add(server_counter)
        
        with self._scheduler_lock:
            found = True
            if workertype not in self._worker_types_list:
                self._worker_types_list.append(workertype)
            if workertype not in self._worker_type_to_worker_id_mapping:
                found = False
                self._worker_type_to_worker_id_mapping[workertype] = []
                self._server_type_to_server_id_mapping[workertype] = []
            self._server_id_to_server_type_mapping[server_counter]
            self._server_type_to_server_id_mapping[workertype].append(server_counter)
            self._server_connections[server_counter] = rpc_client
            self._server_id_counter += 1
            if not found:
                self._priorities[workertype] = {}
                self._deficits[workertype] = {}
                for jobid in self._jobs:
                    self._steps_run_so_far[jobid][workertype] = 0
                    self._job_time_so_far[jobid][workertype] = (0)
                    self._worker_time_so_far[workertype] = 0.0
                    self._throughputs[jobid][workertype] = self._oracle_throughputs[workertype][(self._jobs[jobid].job_type,self._jobs[jobid].scale_factor)]
            
            
            per_worker_ids = []
            for i in range(numgpus):
                workerid = self._worker_id_counter
                per_worker_ids.append(workerid)
                self._worker_ids.append(workerid)
                self._worker_id_counter += 1
                self._worker_types.add(workertype)
                self._cumulative_worker_time_so_far[workerid] = 0.0
                self._worker_id_to_worker_type_mapping[workerid] = workertype
                self._logger.info('增加服务器到集群调度中去,本次增加GPUs序号为:{0}'.format(workerid))
                self._available_worker_ids.put(workerid)
                if workertype not in self._cluster_spec:
                    self._cluster_spec[workertype] = 0
                self._cluster_spec[workertype] += 1
                self._worker_connections[workerid] = rpc_client
                self._worker_start_times[workerid] = self.get_current_timestamp()
            self._worker_type_to_worker_id_mapping[workertype].append(per_worker_ids)
            self._need_to_update_allocation = True
            self._scheduler_cv.notifyAll()
        return (per_worker_ids,self._time_per_iteration)


class SchedulerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        scheduler = self.extra["scheduler"]
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