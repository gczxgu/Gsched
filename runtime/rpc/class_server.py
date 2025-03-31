import ipaddress
import time,grpc,logging,os,sys,socket,traceback
from concurrent import futures

sys.path.append('./runtime/rpc_stubs/')
# print(sys.path)
import worker_to_class_pb2 as w2c_pb2
import worker_to_class_pb2_grpc as w2c_pb2_grpc
import iterator_to_class_pb2 as i2c_pb2
import iterator_to_class_pb2_grpc as i2c_pb2_grpc
import common_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ClassRpcServer(w2c_pb2_grpc.WorkerToClassServicer):   
    def __init__(self,callbacks,logger):
        self._callbacks = callbacks
        self._logger = logger
            
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
            
class ClassIteratorRpcServer(i2c_pb2_grpc.IteratorToClassServicer):
    def __init__(self,callbacks,logger):
        self._callbacks = callbacks
        self._logger = logger
        
    def InitJob(self,request,context):
        job_id = request.job_id
        self._logger.info(
            "对任务进行初始化,任务序号为： {0}".format(job_id))
        init_job_callback = self._callbacks["InitJob"]
        max_steps, max_duration, extra_time = init_job_callback(job_id=job_id)
        if max_steps>0 and max_duration>0:
            self._logger.info(f"任务 {job_id}初始化结果, 最大迭代次数: {max_steps},最大持续时间: {max_duration},额外时间: {extra_time} ")
        else:
            self._logger.error(f"任务 {job_id}初始化失败!,最大迭代次数: {max_steps},最大持续时间: {max_duration},额外时间: {extra_time}")
        return i2c_pb2.UpdateLeaseResponse(max_steps=max_steps,max_duration=max_duration,extra_time=extra_time,)

    def UpdateLease(self, request, context):
        job_id = request.job_id
        self._logger.info(
            f"任务 {job_id}需要对机器 {request.worker_id}续约请求, 已经计算迭代 {request.steps}, 已经工作时间 {request.duration}, "
            f"最大迭代次数 {request.max_steps}, 最大持续时间 {request.duration}")
        update_lease_callback = self._callbacks["UpdateLease"]
        try:
            (
                max_steps,
                max_duration,
                run_time_so_far,
                deadline,
            ) = update_lease_callback(
                job_id,
                request.worker_id,
                request.steps,
                request.duration,
                request.max_steps,
                request.max_duration,
            )
            self._logger.info(
                f"机器 {request.worker_id} 发送新的续约给任务 {job_id},最大迭代次数为 {max_steps},最大持续时间为 {max_duration}"
                f"作业迄今为止运行时间 {run_time_so_far},最大运行时间(deadline) {deadline}")
        except Exception as e:
            self._logger.error(
                f"由于错误 {str(e)} \n 无法给作业 {job_id}进行续约")
            max_steps = request.max_steps
            max_duration = request.max_duration

        return i2c_pb2.UpdateLeaseResponse(
            max_steps=max_steps,
            max_duration=max_duration,
            run_time_so_far=run_time_so_far,
            deadline=deadline,)
    
    
    
def server(port,callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
    )
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor())
    w2c_pb2_grpc.add_WorkerToClassServicer_to_server(
        ClassRpcServer(callbacks, logger), server
    )
    i2c_pb2_grpc.add_IteratorToClassServicer_to_server(
        ClassIteratorRpcServer(callbacks, logger), server
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