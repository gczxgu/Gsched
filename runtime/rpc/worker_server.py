from concurrent import futures
from ipaddress import ip_address
import ipaddress
import socket
import grpc,logging,os,sys,threading,time

import class_to_worker_pb2 as c2w_pb2
import class_to_worker_pb2_grpc as c2w_pb2_grpc
import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
import enums_pb2
import job
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class workerserver(s2w_pb2_grpc.SchedulerToWorkerServicer):
    def __init__(self,callbacks, condition, logger):
        self._callbacks = callbacks
        self._condition = condition
        self._logger = logger

    def RunClass(self, request,context):
        self._logger.debug(f'从服务端收到在该worker上运行class服务器的请求')
        run_class_callback = self._callbacks['RunClass']
        define_dispatcher = self._callbacks['DEFINE']
        define_dispatcher(request.classip,request.classport)
        run_class_callback(request.policy,request.workerids,request.jobids,request.ips,request.ports,request.describes,request.clusterinfos,\
                            request.classip,request.classport)
        return common_pb2.Empty()  


class WorkerServer(c2w_pb2_grpc.ClassToWorkerServicer):
    def __init__(self, callbacks, condition, logger):
        self._callbacks = callbacks
        self._condition = condition
        self._logger = logger
        
    def RunJob(self, request, context):
        self._logger.debug(f"从服务端收到作业运行的请求,其作业描述为 {request.job_descriptions}")
        jobs = []
        for job_description in request.job_descriptions:
            jobs.append(job.Job.from_proto(job_description))
        run_job_callback = self._callbacks["RunJob"]
        run_job_callback(jobs, request.worker_id, request.round_id)
        return common_pb2.Empty()
    
    def KillJob(self, request, context):
        self._logger.debug(f"从服务端收到作业停止的请求,其作业id为 {request.job_id}")
        kill_job_callback = self._callbacks["KillJob"]
        kill_job_callback(request.job_id)
        return common_pb2.Empty()

    def Reset(self, request, context):
        reset_callback = self._callbacks["Reset"]
        reset_callback()
        return common_pb2.Empty()

    def Shutdown(self, request, context):
        shutdown_callback = self._callbacks["Shutdown"]
        shutdown_callback()
        self._condition.acquire()
        self._condition.notify()
        self._condition.release()
        return common_pb2.Empty()

def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
    )
    logger.addHandler(ch)
    condition = threading.Condition()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    c2w_pb2_grpc.add_ClassToWorkerServicer_to_server(
        WorkerServer(callbacks, condition, logger), server
    )
    s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(
        workerserver(callbacks,condition,logger), server
    )
    logger.info("Starting server at port {0}".format(port))
    ip_address = socket.gethostbyname(socket.gethostname())
    # ip_address = '192.168.43.199'
    # server.add_insecure_port("[::]:%d" % (port))
    server.add_insecure_port("%s:%d" % (ip_address, port))
    server.start()
    # Wait for worker server to receive a shutdown RPC from scheduler.
    print(condition)
    with condition:
        condition.wait()
    # Wait for shutdown message to be sent to scheduler.
    time.sleep(5)