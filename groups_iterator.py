import atexit,datetime,json,logging,random,os,time,torch
from cgi import print_form
from filelock import FileLock
from collections.abc import Iterable
from torch.utils.data.dataloader import DataLoader
from runtime.rpc import iterator_client


logging.getLogger("filelock").setLevel(logging.INFO)
random.seed(6666)
INFINITY = 1e9
LEASE_UPDATE_FRACTION = 0.95
LOG_FORMAT = "[{asctime}] [{event}] [{status}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

dataset_len = {
    "CIFAR-10": 50000,
    "ImageNet": 100000,
    "Multi30k": 10000,
    "Wikitext-2": 59675,
    "ML-20M": 117907,
    "Pong": 4,
    "monet2photo": 6287,
}

class GroupIterator:
    def __init__(
        self,
        data_loader,
        checkpoint_dir,
        load_checkpoint_func,
        save_checkpoint_func,
        # if_init ,
        synthetic_data=False,
        write_on_close=True,
        verbose=True,
    ):
        if not isinstance(data_loader, Iterable):
            raise ValueError(
                "Data is of uniterable " "type %s" % (type(data_loader))
            )
        else:
            self._data_loader = data_loader
        
        # try:
        #     for batchidx,batchsize in enumerate(data_loader):
        #         print(batchidx,batchsize)
        #         if batchidx ==0 :
        #             self.batchsize = len(batchsize[0])
        #             break
        # except Exception as e:
        #     print(str(e))
        
        self._write_on_close = write_on_close
        atexit.register(self._close_file_handler)
        if self._write_on_close:
            atexit.register(self._write_info)
        
        self._verbose = verbose
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func
        self._job_id = int(os.environ["GAVEL_JOB_ID"])
        self._worker_id = int(os.environ["GAVEL_WORKER_ID"])
        self._round_id = int(os.environ["GAVEL_ROUND_ID"])
        self._sched_addr = os.environ["GAVEL_SCHED_ADDR"]
        self._sched_port = int(os.environ["GAVEL_SCHED_PORT"])
        self._lock_file = os.path.join(checkpoint_dir, "gavel.lock")
        self._lock = FileLock(self._lock_file)
        # self._gavel_dir = os.path.join(checkpoint_dir, "groups")
        self._round_dir = os.path.join(
            checkpoint_dir, "round={0}".format(self._round_id))
        self.iteras = 0
        # self._worker_dir = os.path.join(
        #     self._round_dir, "worker={0}".format(self._worker_id)
        # )
        
        with self._lock:
            # if not os.path.isdir(self._gavel_dir):
            #     os.mkdir(self._gavel_dir)
            if not os.path.isdir(self._round_dir):
                os.mkdir(self._round_dir)
        self._log_file = os.path.join( 
            self._round_dir, "worker={0}.log".format(self._worker_id)
        )
        self._logger = logging.getLogger("groups_iterator")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)
        self._rpc_client = iterator_client.IteratorRpcClient(
            self._job_id,
            self._worker_id,
            self._sched_addr,
            self._sched_port,
            self._logger,
        )
        self._steps = 0
        self._duration = 0
        self._synthetic_data = synthetic_data
        self._done = False
        if self._synthetic_data:
            self._initial_val = None
        self._lease = Lease(0, 0)
        self._logger.info('qwertyu')
        self._update_lease(init=True)
        self._write_info()
        self._prev_time = time.time() 
    
    def _close_file_handler(self):
        self._logger.removeHandler(self._file_handler)
        self._file_handler.close()
    
    def _write_info(self):
        self._logger.info(
            "{0}".format(self._steps),
            extra={"event": "PROGRESS", "status": "STEPS"},
        )
        self._logger.info(
            "{0}".format(self._duration),
            extra={"event": "PROGRESS", "status": "DURATION"},
        )
    
    def __iter__(self):
        self._iterator = iter(self._data_loader)
        return self
    
    def __next__(self):
        # Update the elapsed time.
        cur_time = time.time()
        if self._prev_time is None:
            self._prev_time = cur_time
        elapsed_time = cur_time - self._prev_time
        self._duration += elapsed_time
        self._prev_time = cur_time

        # if self.iteras >=5 and self.iteras <= 10:
        #     self._logger.info("{0},{1}".format(elapsed_time,self.batchsize),extra={"event": "AI", "status": "datas"})   
        self.iteras += 1     
        
        # Update the lease if necessary.
        # print(f'qqqqqq{self._steps_until_next_lease_update},{self._time_until_next_lease_update}')
        # 每次迭代前先检查是否有足够的时间或需要训练的情况进行,若没有查询能否进行续约
        if (
            self._steps_until_next_lease_update <= 0
            or self._time_until_next_lease_update <= 0
        ):
            self._update_lease()

        # Check if the lease has expired.
        lease_expired = (
            self._duration >= self._lease.max_duration
            or self._steps >= self._lease.max_steps
        )
        
        remain_time = self._lease.max_duration-self._duration
        # if remain_time >1 and lease_expired ==True:
        #     lease_expired = False
        if self._lease.max_steps >0:
            if lease_expired ==False and abs(self._steps - self._lease.max_steps) <=5:
                lease_expired =False
        
        if lease_expired:
            # FIXME: Sometimes the lease does not get received by the gavel iterator
            print(
                f"[{datetime.datetime.now()}] 停止作业的继续训练,self.done标志设置为true"
            )
            self._done = True
            self._logger.info(
                "{0} / {1} steps, {2:.4f} / {3:.4f} seconds".format(
                    self._steps,
                    self._lease.max_steps,
                    self._duration,
                    self._lease.max_duration,
                ),
                extra={"event": "LEASE", "status": "EXPIRED"},
            )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise StopIteration

        # Return a new data item if one exists.
        try:
            if self._synthetic_data and self._initial_val is not None:
                val = self._initial_val
            else:
                val = next(self._iterator)
                if self._synthetic_data and self._initial_val is None:
                    self._initial_val = val
            self._steps += 1
        except StopIteration as e:
            self._write_info()
            raise StopIteration

        if self._synthetic_data and self._steps % len(self._data_loader) == 0:
            raise StopIteration
        

        self._steps_until_next_lease_update -= 1
        self._time_until_next_lease_update -= elapsed_time

        return val

    def __len__(self):
        return len(self._data_loader)

    @property
    def done(self):
        return self._done

    def complete(self, timeout=False):
        timeout_triggered_str = (
            ", triggered by timeout mechanism" if timeout else ""
        )
        print(
            f"[{datetime.datetime.now()}] Setting self._done to True in complete{timeout_triggered_str}"
        )
        self._done = True
        if not self._write_on_close:
            self._write_info()
        self._logger.info("", extra={"event": "LEASE", "status": "COMPLETE"})

    def load_checkpoint(self, *args, **kwargs):
        self._logger.info(
            "", extra={"event": "LOAD CHECKPOINT", "status": "BEGIN"}
        )
        checkpoint = self._load_checkpoint_func(*args, **kwargs)
        self._logger.info(
            "", extra={"event": "LOAD CHECKPOINT", "status": "END"}
        )
        return checkpoint

    def save_checkpoint(self, *args, **kwargs):
        self._logger.info(
            "", extra={"event": "SAVE CHECKPOINT", "status": "BEGIN"}
        )
        retval = self._save_checkpoint_func(*args, **kwargs)
        self._logger.info(
            "", extra={"event": "SAVE CHECKPOINT", "status": "END"}
        )
        return retval

    def _init_logger(self):
        self._logger = logging.getLogger("gavel_iterator")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style="{")
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)

    def _update_lease(self, init=False):
        if init:
            print(
                f"[{datetime.datetime.now()}] GroupIterator initializing training"
            )
            (
                updated_max_steps,
                updated_max_duration,
                extra_time,
            ) = self._rpc_client.init()
            print(
                f"[{datetime.datetime.now()}] GroupIterator initialized training, got {(updated_max_steps, updated_max_duration, extra_time)}"
            )
        else:
            # self._logger.debug("Progress: steps={0}, duration={1}".format(
            #         self._steps, self._duration
            #     ),
            #     extra={"event": "LEASE", "status": "DEBUG"},
            # )
            print(
                f"[{datetime.datetime.now()}] GavelIterator asking for updated lease"
            )
            (
                updated_max_steps,
                updated_max_duration,
                run_time_so_far,
                deadline,
            ) = self._rpc_client.update_lease(
                self._steps,
                self._duration,
                self._lease.max_steps,
                self._lease.max_duration,
            )
            print(
                f"[{datetime.datetime.now()}] GavelIterator received updated lease: {(updated_max_steps, updated_max_duration, run_time_so_far, deadline)}"
            )
            print(
                f"[{datetime.datetime.now()}] self._duration: {self._duration}, run_time_so_far: {run_time_so_far}, deadline: {deadline}"
            )
            extra_time = 0

            # if job is already running over time (due to fluctuating throughput, inter-job interference, etc.),
            # manually mark the job as complete and remove it from Gavel
            if self._duration + run_time_so_far > deadline:
                # job is running over time
                # invoke done_callback, remove the job from Gavel
                print(
                    f"[{datetime.datetime.now()}] projected run time ({self._duration + run_time_so_far}) > deadline ({deadline}), completing job & removing from Gavel"
                )
                self.complete(timeout=True)
                raise StopIteration

        # 代表作业已经训练完了
        if updated_max_steps == self._lease.max_steps:
            self._steps_until_next_lease_update = INFINITY
        else:
            additional_lease_steps = updated_max_steps - self._lease.max_steps
            steps_left_on_current_lease = self._lease.max_steps - self._steps
            self._steps_until_next_lease_update = (
                steps_left_on_current_lease
                + additional_lease_steps 
            )

        if updated_max_duration <= self._lease.max_duration:
            self._time_until_next_lease_update = INFINITY
        else:
            additional_lease_time = (
                updated_max_duration - self._lease.max_duration
            )
            time_left_on_current_lease = (
                self._lease.max_duration - self._duration
            )
            self._time_until_next_lease_update = (
                time_left_on_current_lease
                + additional_lease_time * LEASE_UPDATE_FRACTION
                + extra_time
            )
            self._logger.debug(
                "已经完成了的: steps={0}, duration={1}".format(
                    self._steps, self._duration
                ),
                extra={"event": "LEASE", "status": "DEBUG"},
            )
            self._logger.debug(
                "先前的租约: max_steps={0}, max_duration={1}".format(
                    self._lease.max_steps, self._lease.max_duration
                ),
                extra={"event": "LEASE", "status": "DEBUG"},
            )
            self._logger.debug(
                "这次加的租约: max_steps={0}, max_duration={1}, "
                "extra_time={2}".format(
                    updated_max_steps, updated_max_duration, extra_time
                ),
                extra={"event": "LEASE", "status": "DEBUG"},
            )
            self._logger.debug(
                "做完更新的迭代次数Steps until next lease update={0}".format(
                    self._steps_until_next_lease_update
                ),
                extra={"event": "LEASE", "status": "DEBUG"},
            )
            self._logger.debug(
                "下次进行更新的时间Time until next lease update={0}".format(
                    self._time_until_next_lease_update
                ),
                extra={"event": "LEASE", "status": "DEBUG"},
            )

        # Update the lease.
        self._lease.max_steps = updated_max_steps
        self._lease.max_duration = updated_max_duration + extra_time

























class Lease:
    def __init__(self, max_steps, max_duration):
        self._max_steps = max_steps
        self._max_duration = max_duration

    def __str__(self):
        return "max_steps: %d, max_duration: %f" % (
            self._max_steps,
            self._max_duration,
        )

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, steps):
        self._max_steps = steps

    @property
    def max_duration(self):
        return self._max_duration

    @max_duration.setter
    def max_duration(self, duration):
        self._max_duration = duration