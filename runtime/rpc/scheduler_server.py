import ipaddress
import time,grpc,logging,os,sys,socket,traceback
from concurrent import futures

sys.path.append('./runtime/rpc_stubs/')
# print(sys.path)
import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

import common_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self,callbacks,logger):
        self._callbacks = callbacks
        self._logger = logger
        
    def RegisterWorker(self,request,context):
        register_worker_callback = self._callbacks["RegisterWorker"]
        try:
            worker_ids, round_duration = register_worker_callback(
                workertype=request.worker_type,
                numgpus=request.num_gpus,
                ip_addr=request.ip_addr,
                port=request.port,
            )
            self._logger.info(
                f"成功登记机器类型为 {request.worker_type},其id记录为 {str(worker_ids)}")
            return w2s_pb2.RegisterWorkerResponse(
                success=True,
                worker_ids=worker_ids,
                round_duration=round_duration,
            )
        except Exception as e:
            self._logger.error(f"发生错误 {e}无法登记机器")
            return w2s_pb2.RegisterWorkerResponse(
                successful=False, error_message=e)
            
    def Done(self, request, context):
        done_callback = self._callbacks["Done"]
        try:
            job_id = request.job_id
            self._logger.info(f"任务 {job_id}收到完成通知,其使用的机器id {request.worker_id},完成迭代次数为 {str(request.num_steps)},执行时间 {str(request.execution_time)} ")
            done_callback(
                job_id,
                request.worker_id,
                request.num_steps,
                request.execution_time,
                request.iterator_log,
            )
        except Exception as e:
            self._logger.error(f"由于错误{str(e)}无法处理作业 {job_id}的完成请求")
            traceback.print_exc()
        return common_pb2.Empty()
                
    
    
def server(port,callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
    )
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor())
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
        SchedulerRpcServer(callbacks, logger), server
    )

    ip_address = socket.gethostbyname(socket.gethostname())
    # ip_address = '192.168.43.199'
    server.add_insecure_port("%s:%d" % (ip_address, port))
    logger.info("Starting server at {0}:{1}".format(ip_address, port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)