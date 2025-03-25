import os,sys,argparse,datetime,queue,time,pickle
import utils
import scheduler


SLEEP_TIME = 10
def runnning(args):
    # trace信息
    trace_name = os.path.splitext(os.path.basename(args.trace_file))[0]
    trace_dir = os.path.dirname(args.trace_file)
    # 队列信息
    # 初始化的pickle记录了不同训练的指标（duration、mem、utils）
    throughputfile = './throughputs/tacc_throughputs.json'
    jobs,arrival_time = utils.generate_pickle_file(args.trace_file,throughputfile)
    job_queue = queue.Queue()
    jobs_to_complete = set()
    for i in range(len(jobs)):
        jobs_to_complete.add(i)
    for (job,arrival_time) in zip(jobs,arrival_time):
        job_queue.put((job,arrival_time))
    
    policy = utils.get_policy(args.policy,args.solver,args.seed)
    pathfile = os.path.dirname(__file__)
    pickle_filepath = os.path.join(os.path.join(pathfile,f'logging'),f'{trace_name}.pickle')
    
    sched = scheduler.Scheduler(policy,
                             seed = args.seed,
                             throughputs_file= args.throughputs_file,
                             expected_num_workers=args.expected_num_workers,
                             max_rounds=args.max_rounds,
                             pickle_file=pickle_filepath,
                             time_per_iteration=args.time_per_iteration,
                             )
    with open(pickle_filepath,'rb') as f:
        trace_pickle = pickle.load(f)
        job_id = 0
        try:
            start_time = datetime.datetime.now()
            print(f"--"*50)
            print(f"调度 START北京时间:{start_time}")
            while(not sched.is_done(jobs_to_complete) and
                  not job_queue.empty()) :
                job,arrival_time = job_queue.get()
                job.duration = sum(trace_pickle[job_id]['duration_every_epoch'])
                job_id +=1
                utils.read_all_throughputs_json(throughputfile) 
                while True:
                    current_time = datetime.datetime.now()
                    elapsed_seconds = (current_time - start_time).seconds
                    remaining_time = arrival_time - elapsed_seconds
                    if remaining_time<=0:
                        sched.add_job(job)
                        break
                    elif sched.is_done(jobs_to_complete):
                        break
                    else:
                        time.sleep(SLEEP_TIME)
                if sched.is_done(jobs_to_complete):
                    break
            while not sched.is_done(jobs_to_complete):
                    time.sleep(SLEEP_TIME)
        except KeyboardInterrupt as e:
            print('输入了crtl+c导致调度结束')






if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=['fifo','sjf'],
                        help='Scheduler policy')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机数种子')
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver求解器')
    parser.add_argument('-th','--throughputs_file', type=str,
                        # default=None,
                        default = r'./throughputs/actual_throughput_new.json',
                        # default = r'./throughputs/tacc_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('--expected_num_workers', type=int, default=None,
                        help='Total number of workers expected')
    parser.add_argument('--time_per_iteration','-iter', type=int, default=20,
                        help='每一轮的时间')              
    parser.add_argument('-s', '--window-start', type=int, default=None,
                        help='measurement window start (job id)')
    parser.add_argument('-e', '--window-end', type=int, default=None,
                        help='Measurement window end (job ID)')
    parser.add_argument('--max_rounds', type=int, default=None,
                        help='Maximum number of rounds to run')
    parser.add_argument('--timeline_dir', type=str, default=None,
                        help='Directory to save timelnes to')
    parser.add_argument("--pickle_output_dir",
        type=str,default="../../pickle_output",help=".pickle文件存储位置",)
    runnning(parser.parse_args())