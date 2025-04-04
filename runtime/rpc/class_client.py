import grpc,logging,os,sys,time
sys.path.append('./runtime/rpc_stubs')

import class_to_worker_pb2 as c2w_pb2
import class_to_worker_pb2_grpc as c2w_pb2_grpc
import common_pb2

LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ClassRpcClient:
    def __init__(self, server_ip_addr, port):
        self._addr = server_ip_addr
        self._port = port
        self._server_loc = "%s:%d" % (server_ip_addr, port)
    
    @property
    def addr(self):
        return self._addr
    
    @property
    def port(self):
        return self._port
    
    def run_job(self, job_descriptions, worker_id, round_id):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = c2w_pb2_grpc.ClassToWorkerStub(channel)
            request = c2w_pb2.RunJobRequest()
            for (
                job_id,
                command,
                working_directory,
                needs_data_dir,
                num_steps_arg,
                num_steps,
                mode,
                mps_thread_percentage,
            ) in job_descriptions:
                job_description = request.job_descriptions.add()
                job_description.job_id = job_id  # job_id is a JobIdPair
                job_description.command = command
                job_description.working_directory = working_directory
                job_description.needs_data_dir = needs_data_dir
                job_description.num_steps_arg = num_steps_arg
                job_description.num_steps = num_steps
                job_description.mode = mode
                job_description.mps_thread_percentage = mps_thread_percentage
            request.worker_id = worker_id
            request.round_id = round_id
            response = stub.RunJob(request)
    
    def kill_job(self, job_id):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = c2w_pb2_grpc.ClassToWorkerStub(channel)
            request = c2w_pb2.KillJobRequest()
            request.job_id = job_id  # job_id is a JobIdPair
            response = stub.KillJob(request)
    
    def reset(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = c2w_pb2_grpc.ClassToWorkerStub(channel)
            response = stub.Reset(common_pb2.Empty())
    
    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = c2w_pb2_grpc.ClassToWorkerStub(channel)
            response = stub.Shutdown(common_pb2.Empty())
    
    