import grpc,logging,os,sys,time

sys.path.append('./runtime/rpc_stubs')

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import worker_to_class_pb2 as w2c_pb2
import worker_to_class_pb2_grpc as w2c_pb2_grpc

LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class WorkerRpcClient:
    def __init__(self,worker_type,
        worker_ip_addr,
        worker_port,
        sched_ip_addr,
        sched_port,):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
        )
        logger.addHandler(ch)
        self._logger = logger
        self._worker_type = worker_type
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_ip_addr = sched_ip_addr
        self._sched_port = sched_port
        self._sched_loc = "%s:%d" % (sched_ip_addr, sched_port)
        
    def register_worker(self, num_gpus):
        request = w2s_pb2.RegisterWorkerRequest(
            worker_type=self._worker_type,
            ip_addr=self._worker_ip_addr,
            port=self._worker_port,
            num_gpus=num_gpus,
        )
        with grpc.insecure_channel(self._sched_loc) as channel:
            self._logger.debug(f"开始尝试登陆机器,其ip地址为 {self._worker_ip_addr},端口号为 {self._worker_port}")
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            
            response = stub.RegisterWorker(request)
            if response.success:
                self._logger.info(f'成功登记机器id为 {str(response.worker_ids)},每轮迭代时间 {response.round_duration}')
                return (response.worker_ids, response.round_duration, None)
            else:
                assert response.HasField("error")
                self._logger.error(f"由于出现错误{response.error}登记失败")
                return (None, response.error)
    
    

class workerrpcclient:
    def __init__(self,worker_type,
        worker_ip_addr,
        worker_port,
        sched_ip_addr,
        sched_port,):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
        )
        logger.addHandler(ch)
        self._logger = logger
        self._worker_type = worker_type
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_ip_addr = sched_ip_addr
        self._sched_port = sched_port
        self._sched_loc = "%s:%d" % (sched_ip_addr, sched_port)

    def notify_scheduler(self, worker_id, job_descriptions):
        request = w2c_pb2.DoneRequest()
        request.worker_id = worker_id
        for job_description in job_descriptions:
            request.job_id.append(job_description[0])
            request.execution_time.append(job_description[1])
            request.num_steps.append(job_description[2])
            request.iterator_log.append(job_description[3])
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = w2c_pb2_grpc.WorkerToClassStub(channel)
            try:
                response = stub.Done(request)
            except Exception as e:
                print(str(e))
            job_ids = [job_description[0] for job_description in job_descriptions]
            self._logger.info(
                    f'通知调度器作业 {job_ids}已经完成')