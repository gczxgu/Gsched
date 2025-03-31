import grpc,logging,os,sys,time
sys.path.append('./runtime/rpc_stubs')

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2
from scheduler_to_worker_pb2 import JobsIds,Ips

LOG_FORMAT = "{name}:{levelname} [{asctime}] {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SchedulerRpcClient:
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
    
    def run_class(self,policy,workerids,jobsid,ips,ports,job_descriptions,clusterinfos,classip,classport):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.RunClassRequest()
            request.policy = policy
            classinfos_instance = JobsIds()
            # for i in workerids:
            #     classinfos_instance.ids.append(i)
            classinfos_instance.ids.extend(workerids)
            request.workerids.append(classinfos_instance)
            jobsids_instance = JobsIds()
            jobsids_instance.ids.extend(jobsid)
            request.jobids.append(jobsids_instance)
            ips_instance = Ips()
            ips_instance.ips.extend(ips)
            request.ips.append(ips_instance)
            ports_instance = JobsIds()
            ports_instance.ids.extend(ports)
            request.ports.append(ports_instance)
            for (
                job_id,
                job_type,
                command,
                working_directory,
                needs_data_dir,
                num_steps_arg,
                num_steps,
                mode,
                mps_thread_percentage,
                scale_factor,
            ) in job_descriptions:
                job_description = request.describes.add()
                job_description.job_id = job_id  # job_id is a JobIdPair
                job_description.job_type = job_type
                job_description.command = command
                job_description.working_directory = working_directory
                job_description.needs_data_dir = needs_data_dir
                job_description.num_steps_arg = num_steps_arg
                job_description.num_steps = num_steps
                job_description.mode = mode
                job_description.mps_thread_percentage = mps_thread_percentage
                job_description.scale_factor = scale_factor
            for(type,indexs,nums) in clusterinfos:
                cluster_info = request.clusterinfos.add()
                cluster_info.type = type
                cluster_info.indexs = indexs    
                cluster_info.nums = nums
            request.classip = classip
            request.classport = classport
            try:
                response = stub.RunClass(request)
            except grpc.RpcError as e:
                print(f"gRPC error: {e.code()} - {e.details()}")
    
    def run_job(self, job_descriptions, worker_id, round_id):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.RunJobRequest()
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
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.KillJobRequest()
            request.job_id = job_id  # job_id is a JobIdPair
            response = stub.KillJob(request)
    
    def reset(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            response = stub.Reset(common_pb2.Empty())
    
    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            response = stub.Shutdown(common_pb2.Empty())
    
    