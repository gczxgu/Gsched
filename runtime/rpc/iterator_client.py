import grpc,sys,os

path1 = os.path.dirname(__file__)
path2 = os.path.dirname(path1)
path3 = os.path.join(path2,'rpc_stubs')
# print(path3)
sys.path.append(path3)
sys.path.append('./runtime/rpc_stubs')
sys.path.append('../rpc_stubs')

import iterator_to_class_pb2 as i2s_pb2
import iterator_to_class_pb2_grpc as i2s_pb2_grpc

class IteratorRpcClient:
    def __init__(self, job_id, worker_id, sched_ip_addr, sched_port, logger):
        self._job_id = job_id
        self._worker_id = worker_id
        self._sched_loc = "%s:%d" % (sched_ip_addr, sched_port)
        self._logger = logger
    
    def init(self):
        request = i2s_pb2.InitJobRequest(job_id=self._job_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToClassStub(channel)
            try:
                self._logger.info(
                    "", extra={"event": "INIT", "status": "REQUESTING"})
                response = stub.InitJob(request)
                self._logger.info(f'初始化迭代次数为 {response.max_steps},初时化持续时间为 {response.max_duration}')
                if response.max_steps > 0 and response.max_duration > 0:
                    self._logger.info(
                        "Initial lease: max_steps {0}, "
                        "max_duration={1:.4f}".format(
                            response.max_steps, response.max_duration
                        ),
                        extra={"event": "INIT", "status": "COMPLETE"},
                    )
                    self._logger.info(
                        "{0}".format(
                            response.max_steps
                        ),
                        extra={"event": "INIT", "status": "STEPS"}
                    )
                    self._logger.info(
                        "{0}".format(
                            response.max_duration
                        ),
                        extra={"event": "INIT", "status": "DURATION"}
                    )
                else:
                    self._logger.error(
                        "", extra={"event": "INIT", "status": "FAILED"}
                    )
                return (
                    response.max_steps,
                    response.max_duration,
                    response.extra_time,
                )
            except grpc.RpcError as e:
                self._logger.error(
                    "{0}".format(e), extra={"event": "INIT", "status": "ERROR"}
                )
            return (0, 0, 0)

    def update_lease(self, steps, duration, max_steps, max_duration):
        # 
        request = i2s_pb2.UpdateLeaseRequest(
            job_id=self._job_id,
            worker_id=self._worker_id,
            steps=steps,
            duration=duration,
            max_steps=max_steps,
            max_duration=max_duration,
        )
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = i2s_pb2_grpc.IteratorToClassStub(channel)
            self._logger.info(
                "", extra={"event": "LEASE", "status": "REQUESTING"}
            )
            try:
                response = stub.UpdateLease(request)
                self._logger.info(
                    "New lease: max_steps={0}, max_duration={1:.4f},".format(
                        response.max_steps, response.max_duration
                    ),
                    extra={"event": "LEASE", "status": "UPDATED"},
                )
                return (
                    response.max_steps,
                    response.max_duration,
                    response.run_time_so_far,
                    response.deadline,
                )
            # except grpc.RpcError as e:
            except Exception as e:
                self._logger.error(
                    "{0}".format(e),
                    extra={"event": "LEASE", "status": "ERROR"},
                )
        # return (max_steps, max_duration)
        return (max_steps, max_duration, 0, 0)   





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