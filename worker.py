import argparse,os,signal,sys,threading,socket,logging,datetime,time,traceback
from pydoc import describe

import utils
from runtime.rpc import worker_client,worker_server,dispatcher
from runtime.rpc import iterator_client
from class_min import CLASSmin
import class_min
import job


CHECKPOINT_DIR_NAME = "checkpoints"
LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class Worker:
    def __init__(self,worker_type,
        sched_addr,
        sched_port,
        worker_port,
        num_gpus,
        static_run_dir,
        accordion_run_dir,
        gns_run_dir,
        data_dir,
        checkpoint_dir,
        use_mps,):
        self.static_run_dir = static_run_dir
        self.accordion_run_dir = accordion_run_dir
        self.gns_run_dir = gns_run_dir
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.mps = use_mps
        self._start_time = time.time()
        logger = logging.getLogger(__name__)
        logging_level_dict = {"INFO":logging.INFO,"DEBUG":logging.DEBUG}
        logger.setLevel(logging_level_dict["DEBUG"])
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT,style="{"))
        logger.addHandler(ch)
        logger.addHandler(logging.FileHandler(os.path.join(os.path.join(os.path.dirname(__file__),'logging'),f'worker_output.txt'),mode='w',))
        self._logger = WorkerAdapter(logger,{"worker": self, "start_timestamp": datetime.datetime.now()})
        self._logging_handler = ch
        
        num_available_gpus = utils.get_num_gpus()
        if num_gpus > num_available_gpus:
            raise ValueError(f'需要 {num_gpus}个硬件来使用,但是目前总共只有 {num_available_gpus}个可用')        # signal.signal(signal.SIGINT, self._signal_handler)
        self._gpu_ids = list(range(num_gpus))
        self._worker_type = worker_type
        self._worker_addr = socket.gethostbyname(socket.gethostname())
        # self._worker_addr = '192.168.43.199'
        self._worker_port = worker_port
        self._worker_rpc_client = worker_client.WorkerRpcClient(self._worker_type,self._worker_addr,self._worker_port,sched_addr,sched_port,)
        callbacks = {
            "RunJob": self._run_job_callback,
            "KillJob": self._kill_job_callback,
            "Reset": self._reset_callback,
            "Shutdown": self._shutdown_callback,
            "RunClass": self._run_class_callback,
            "DEFINE":self._define_dispatcher_callback}        
        # 守护进程
        
        self._server_thread = threading.Thread(target=worker_server.serve, args=(worker_port, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()
        
        self._round_duration = 20
        # (self._worker_ids,self._round_duration,error,) = \
        # self._worker_rpc_client.register_worker(len(self._gpu_ids))
        # if error:
        #     raise RuntimeError(error)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # self._dispatcher = dispatcher.Dispatcher(self._round_duration,
        #     self._gpu_ids,
        #     self._worker_rpc_client,
        #     sched_addr,
        #     sched_port,
        #     static_run_dir,
        #     accordion_run_dir,
        #     gns_run_dir,
        #     data_dir,
        #     checkpoint_dir,
        #     self._logger,
        #     use_mps=use_mps,)
        
        self._server_thread.join()
    
    def _signal_handler(self,sig,frame):
        self._dispatcher.shutdown()
        print('命令:接收停止信号signal')
        sys.exit(0)
    
    
    def get_current_timestamp(self):
        return time.time() - self._start_time
    
    def _define_dispatcher_callback(self,classip,classport):
        self._worker_rpc_clients_class = worker_client.workerrpcclient(self._worker_rpc_client,self._worker_addr,self._worker_port,classip,50080)
        self._dispatcher = dispatcher.Dispatcher(self._round_duration,
            self._gpu_ids,
            self._worker_rpc_clients_class,
            classip,
            50080,
            self.static_run_dir,
            self.accordion_run_dir,
            self.gns_run_dir,
            self.data_dir,
            self.checkpoint_dir,
            self._logger,
            use_mps=self.mps,)

    def _run_class_callback(self,policy,classinfo,jobsid,ips,ports,jobsdescribes,clusterinfos,classip,classport):
        jobs = []
        for job_description in jobsdescribes:
            jobs.append(job.Job.from_proto(job_description))
        self._logger.info(f'在该服务器上运行classmin,classmin运行地址 {classip}:{50080}')
        try:
            class_min.server(policy,classinfo,jobsid,ips,ports,jobs,clusterinfos)
        except Exception as e:
            self._logger.error(f'发生错误{str(e)}')
            traceback.print_exc()
            sys.exit()

    
    def _run_job_callback(self, jobs, worker_id, round_id):
        self._logger.info(f'当前调度器处于第{round_id}轮 在硬件 {worker_id}上开始启动作业')
        # print("命令:启动作业run_job")
        while True:
            try:
                self._dispatcher
                break
            except Exception as e:
                self._logger.info(f'发生错误 {str(e)}')
                time.sleep(1)
                continue
        self._logger.info(f'开始启动作业进行训练')
        # print("命令:开始工作start_job")
        self._dispatcher.dispatch_jobs(jobs, worker_id, round_id)
    
    def _kill_job_callback(self, job_id):
        self._dispatcher._kill_jobs(job_id=job_id)
        self._logger.info(f'作业因为某种原因被结束')
        print('命令:作业结束kill_job')
    
    def _reset_callback(self):
        self._dispatcher.reset()
    
    def _shutdown_callback(self):
        self._dispatcher.shutdown()
    
    def join(self):
        self._server_thread.join()
        
    def read_loggs(self):
        pass
    

class WorkerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        worker = self.extra["worker"]
        timestamp = worker.get_current_timestamp()
        timestamp = self.extra["start_timestamp"] + datetime.timedelta(
                0, timestamp
            )
        return "[%s] %s" % (timestamp, msg), kwargs  
        


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Run a worker process")
    parser.add_argument(
        "-t", "--worker_type", type=str, required=True,)
    parser.add_argument("-i","--ip_addr",type=str,required=True,
        help="IP address for scheduler server",)
    parser.add_argument("-s","--sched_port",type=int,default=50060,
        help="Port number for scheduler server",)
    parser.add_argument("-w","--worker_port",type=int,default=50071,
        help="Port number for worker server",)
    parser.add_argument("-g","--num_gpus",type=int,default=1,
        help="Number of available GPUs",)
    parser.add_argument("--static_run_dir",type=str,
    # default='/home/nudt/Desktop/simulator/groups_easy/workloads/pytorch/',
        default=r'D:\real_sched\groups_easy\workloads\pytorch',
        help="Directory to run static jobs from",)
    parser.add_argument("--accordion_run_dir",type=str,default='../shockwave-main/accordion_workloads/pytorch/',
        help="Directory to run accordion jobs from",)
    parser.add_argument("--gns_run_dir",type=str,default='../shockwave-main/gns_workloads/pytorch/',
        help="Directory to run gns jobs from",)
    parser.add_argument("--data_dir",type=str,default='./data/',
        help="Directory where data is stored",)
    parser.add_argument("--checkpoint_dir",type=str,
        # default='/home/nudt/Desktop/simulator/groups_easy/checkpoint/models',
        default= r'D:\real_sched\groups_easy\checkpoint\models',
        help="Directory where checkpoints is stored",)
    parser.add_argument("--use_mps",action="store_true",default=False,
        help="If set, enable CUDA MPS",)
    args = parser.parse_args()
    opt_dict = vars(args)
    worker = Worker(
        opt_dict["worker_type"],
        opt_dict["ip_addr"],
        opt_dict["sched_port"],
        opt_dict["worker_port"],
        opt_dict["num_gpus"],
        opt_dict["static_run_dir"],
        opt_dict["accordion_run_dir"],
        opt_dict["gns_run_dir"],
        opt_dict["data_dir"],
        opt_dict["checkpoint_dir"],
        opt_dict["use_mps"],
    )
        