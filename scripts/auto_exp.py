# This python script will automatically run the experiments

# %%

# Import the necessary libraries

import os
import subprocess
from dataclasses import dataclass, field, asdict
from dataclasses import Field as DataclassField
from typing import List, Union, Dict, Optional, Literal, TypedDict
import re
import pandas as pd
from datetime import datetime
from datetime import timezone
import time

import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue
import asyncio
from multiprocessing import Pool
from copy import deepcopy


# Necessary structures

@dataclass
class alg_config:
    use_lazy: Union[None,int] = None  # 0 -> use GOR, 1 -> use LazyDijkstra
    init_kappa: Union[None,int] = None  # 0 -> use number of nodes, 1 -> use infinity
    k_factor: Union[None, int] = None  # by what factor to reduce the number of Dijkstra calls
    rand_label: Union[None,int] = None  # 0 -> label light nodes with Dijkstra, 1 -> label light nodes randomly
    fixdagedges: Union[None,int] = None  # 1 -> normal fixdagedges version, 2 -> old fixdagedges version
    cutedges: Union[None,int] = None  # 1 -> normal cutedges version, 2 -> new cutedges version, 3 -> cut edges in half
    rec_limit: Union[None, int] = None  # recursion limit
    cutedgesseed: Union[None, int] = None  # cut edges seed
    diam_apprx: Union[None,int] = None  # 0 -> normal, 1 -> approximate diameter
    ol_seed: Union[None, int] = None  # out light labeling seed
    il_seed: Union[None, int] = None  # in light labeling seed
    eg_sort_scc: Union[None,int] = None  # 0 -> no edge sorting during SCC, 1 -> edge sorting during SCC

    def to_list(self):
        ret = []
        ret.append(self.use_lazy if self.use_lazy is not None else DEFAULT_ALG_CONFIG.use_lazy)
        ret.append(self.init_kappa if self.init_kappa is not None else DEFAULT_ALG_CONFIG.init_kappa)
        ret.append(self.k_factor if self.k_factor is not None else DEFAULT_ALG_CONFIG.k_factor)
        ret.append(self.rand_label if self.rand_label is not None else DEFAULT_ALG_CONFIG.rand_label)
        ret.append(self.fixdagedges if self.fixdagedges is not None else DEFAULT_ALG_CONFIG.fixdagedges)
        ret.append(self.cutedges if self.cutedges is not None else DEFAULT_ALG_CONFIG.cutedges)
        ret.append(self.rec_limit if self.rec_limit is not None else DEFAULT_ALG_CONFIG.rec_limit)
        ret.append(self.cutedgesseed if self.cutedgesseed is not None else DEFAULT_ALG_CONFIG.cutedgesseed)
        ret.append(self.diam_apprx if self.diam_apprx is not None else DEFAULT_ALG_CONFIG.diam_apprx)
        ret.append(self.ol_seed if self.ol_seed is not None else DEFAULT_ALG_CONFIG.ol_seed)
        ret.append(self.il_seed if self.il_seed is not None else DEFAULT_ALG_CONFIG.il_seed)
        ret.append(self.eg_sort_scc if self.eg_sort_scc is not None else DEFAULT_ALG_CONFIG.eg_sort_scc)
        return ret

    def modify(self, **kwargs):
        modified_instance = deepcopy(self)  # Create a copy of the instance

        for key, value in kwargs.items():
            if hasattr(modified_instance, key):
                setattr(modified_instance, key, value)  # Modify the data member with the new value
            else:
                raise ValueError(f"Invalid key: {key}")

        return modified_instance

DEFAULT_ALG_CONFIG = alg_config(use_lazy=1,
                                init_kappa=0,
                                k_factor=1,
                                rand_label=0,
                                fixdagedges=1,
                                cutedges=1,
                                rec_limit=100,
                                cutedgesseed=1234,
                                diam_apprx=0,
                                ol_seed=1234,
                                il_seed=12134,
                                eg_sort_scc=0)

@dataclass
class exp_config:
    alg_goal: Union[None, Literal["SSSP", "NegCycle"]] = None
    exp_type: Union[None, Literal["time", "check"]] = None
    alg_name: Union[None, Literal["NaiveBFM", "BFCT", "Dijkstra", "GOR", "LazyD", "BCF"]] = None
    filename: Union[None, str] = None
    source: Union[None, int] = None
    reps: Union[None, int] = None

    # If necessary, add controls for NegCycle
    def to_str(self):
        if self.alg_goal is None or self.exp_type is None or self.alg_name is None or self.filename is None or self.source is None:
            raise ValueError("alg_goal, exp_type, alg_name, filename, source cannot be None")
        
        ret = f"{self.alg_goal} {self.exp_type} {self.alg_name} {self.filename} {self.source}"
        if self.reps is None:
            if self.exp_type != "check":
                raise ValueError("reps cannot be None for time experiments")
        elif self.exp_type == "time":
            ret += f" {self.reps}"
        
        return ret

    def to_list(self):
        return [self.alg_goal, self.exp_type, self.alg_name, self.filename, self.source, self.reps]

    def modify(self, **kwargs):
        modified_instance = deepcopy(self)  # Create a copy of the instance

        for key, value in kwargs.items():
            if hasattr(modified_instance, key):
                setattr(modified_instance, key, value)  # Modify the data member with the new value
            else:
                raise ValueError(f"Invalid key: {key}")

        return modified_instance

# Constants
FIRST_ROW_OF_CSV = "timestamp,alg_goal,exp_type,alg_name,filename,source,reps,use_lazy,init_kappa,k_factor,rand_label,fixdagedges,cutedges,rec_limit,cutedgesseed,diam_apprx,ol_seed,il_seed,eg_sort_scc,avg_time,std,min,max,n,m,exp_name\n"


# Helping functions

def read_existing_exp_file(filename : str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    if 'exp_name' not in df:
        df['exp_name'] = 'None'
    return df

def get_exp_file(filename : str) -> pd.DataFrame:
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(FIRST_ROW_OF_CSV)

    return read_existing_exp_file(filename)

def extract_numbers(string):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    numbers = re.findall(pattern, string)
    return numbers

def parseResults(output : str) -> list:
    return [extract_numbers(line) for line in output.split("\n") if line.strip() != ""]


def run_single_exp(conf_alg : alg_config, conf_exp : exp_config, output_file : str = 'tmp.txt', clean_output : bool = True, timeout : Union[None, float] = None) -> list:
    alg_config_str = [f"{name}={value}" for name, value in asdict(conf_alg).items() if value is not None]
    exp_config_str = conf_exp.to_str()

    # Run the experiment
    if alg_config_str == "":
        exp_args = ["./Main", exp_config_str, output_file]
    else:
        exp_args = ["./Main", *alg_config_str, exp_config_str, output_file]

    exp_args_copy = exp_args.copy()
    exp_args_copy[-2] = '"' + exp_args_copy[-2] + '"'
    print(f"Running: {' '.join(exp_args_copy)}")
    try:
        subprocess.run(exp_args, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Timeout expired ({timeout} seconds)")
        return [timeout, '-1', '-1', '-1']
    # print('Done.')

    # Read the output file
    with open(output_file, 'r') as f:
        output = f.read()
    
    if clean_output:
        os.remove(output_file)

    return parseResults(output)[0]


def get_graph_info(filename):
    with open(filename, 'r') as f:
        try:
            n_nodes = int(f.readline().strip())
        except ValueError:
            n_nodes = -1

    n_edges = os.popen(f"wc -l < {filename}").read().strip().split()
    if len(n_edges) > 0:
          n_edges = int(n_edges[0]) - 1

    return n_nodes, n_edges

def new_single_exp(df_results : pd.DataFrame, conf_alg : alg_config, conf_exp : exp_config) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    times = run_single_exp(conf_alg, conf_exp)

    # print('Loading graph info...')
    n, m = get_graph_info(conf_exp.filename)
    # print('Done.')

    # Add the results to the dataframe
    df_results.loc[len(df_results)] = [timestamp] + conf_exp.to_list() + conf_alg.to_list() + times + [n, m]

def new_parallel_exps(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], jobs : int) -> None:

    alg_config_strs = [' '.join([f"{name}={value}" for name, value in asdict(conf_alg).items() if value is not None]) for conf_alg in conf_algs]
    exp_config_strs = [conf_exp.to_str() for conf_exp in conf_exps]
    exps_args = [['./Main', exp_config_str] if alg_config_str == "" else
                ['./Main', alg_config_str, exp_config_str]
                for alg_config_str, exp_config_str in zip(alg_config_strs, exp_config_strs)]
    checked_graphs = {}
    
    jobs = min(jobs, len(exps_args))
    processes = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for i in range(jobs):
        tmp_filename = f'tmp{i}.txt'
        exp_args = exps_args.pop()
        conf_alg = conf_algs.pop()
        conf_exp = conf_exps.pop()
        p = subprocess.Popen(exp_args + [tmp_filename])
        processes.append((p, exp_args, tmp_filename, timestamp, conf_alg, conf_exp))
        
    active_jobs = jobs
    while active_jobs > 0:
        for i, p in enumerate(processes):
            if p is None or p[0].poll() is None: continue

            tmp_filename = p[2]

            # Read the output file
            with open(tmp_filename, 'r') as f:
                output = f.read()
            times = parseResults(output)[0]
            timestamp = p[3]
            conf_alg = p[4]
            conf_exp = p[5]
            graph_filename = conf_exp.filename
            n, m = checked_graphs[graph_filename] if graph_filename in checked_graphs else get_graph_info(graph_filename)
            df_results.loc[len(df_results)] = [timestamp] + conf_exp.to_list() + conf_alg.to_list() + times + [n, m]

            if len(exps_args) > 0:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                exp_args = exps_args.pop()
                conf_alg = conf_algs.pop()
                conf_exp = conf_exps.pop()
                p = subprocess.Popen(exp_args + [tmp_filename])
                processes[i] = (p, exp_args, tmp_filename, timestamp, conf_alg, conf_exp)
            else:
                processes[i] = None
                active_jobs -= 1

        time.sleep(1)
    

def new_single_exp2(conf_alg : alg_config, conf_exp : exp_config) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    times = run_single_exp(conf_alg, conf_exp, 'tmp.txt', False)

    # print('Loading graph info...')
    n, m = get_graph_info(conf_exp.filename)
    # print('Done.')

    # Add the results to the dataframe
    return [timestamp] + conf_exp.to_list() + conf_alg.to_list() + times + [n, m]


def new_parallel_exps2(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], jobs : int):
# def main2(processes, max_threads):
    process_queue = Queue()
    for conf_alg, conf_exp in zip(conf_algs, conf_exps):
        process_queue.put((conf_alg, conf_exp))
    
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_data = {}
        
        # Submit initial batch of tasks
        for _ in range(jobs):
            if not process_queue.empty():
                data = process_queue.get()
                future = executor.submit(new_single_exp2, data[0], data[1])
                future_to_data[future] = data
        
        # As tasks complete, submit new ones
        while future_to_data:
            for future in as_completed(future_to_data):
                data = future_to_data.pop(future)
                try:
                    result = future.result()
                    df_results.loc[len(df_results)] = result
                    # Handle result if needed
                except Exception as exc:
                    print(f'Process {data["command"]} generated an exception: {exc}')
                
                if not process_queue.empty():
                    new_data = process_queue.get()
                    new_future = executor.submit(new_single_exp2, new_data[0][1])
                    future_to_data[new_future] = new_data

def new_parallel_exps3(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], jobs : int):
# def main2(processes, max_threads):
    process_queue = Queue()
    for conf_alg, conf_exp in zip(conf_algs, conf_exps):
        process_queue.put((conf_alg, conf_exp))
    
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_data = {}
        
        # Submit initial batch of tasks
        for _ in range(jobs):
            if not process_queue.empty():
                data = process_queue.get()
                future = executor.submit(new_single_exp2, data[0], data[1])
                future_to_data[future] = data
        
        # As tasks complete, submit new ones
        while future_to_data:
            for future in as_completed(future_to_data):
                data = future_to_data.pop(future)
                try:
                    result = future.result()
                    df_results.loc[len(df_results)] = result
                    # Handle result if needed
                except Exception as exc:
                    print(f'Process {data["command"]} generated an exception: {exc}')
                
                if not process_queue.empty():
                    new_data = process_queue.get()
                    new_future = executor.submit(new_single_exp2, new_data[0][1])
                    future_to_data[new_future] = new_data

async def new_single_exp3(df_results : pd.DataFrame, conf_alg : alg_config, conf_exp : exp_config, output_file : str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    alg_config_str = ' '.join([f"{name}={value}" for name, value in asdict(conf_alg).items() if value is not None])
    exp_config_str = conf_exp.to_str()

    # Run the experiment
    if alg_config_str == "":
        exp_args = ["./Main", exp_config_str, output_file]
    else:
        exp_args = ["./Main", alg_config_str, exp_config_str, output_file]

    exp_args_copy = exp_args.copy()
    exp_args_copy[-2] = '"' + exp_args_copy[-2] + '"'
    print(f"Running: {' '.join(exp_args_copy)}")
    # subprocess.run(exp_args, check=True)
    proc = await asyncio.create_subprocess_exec(*exp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    result_proc = stdout.decode()
    print('Done.', end=' ')

    # Read the output file
    with open(output_file, 'r') as f:
        output = f.read()

    times = parseResults(output)[0]

    # times = run_single_exp(conf_alg, conf_exp)

    # print('Loading graph info...')
    n, m = get_graph_info(conf_exp.filename)
    # print('Done.')

    # Add the results to the dataframe
    result = [timestamp] + conf_exp.to_list() + conf_alg.to_list() + times + [n, m]
    
    df_results.loc[len(df_results)] = result


# async def new_parallel_exps4(processes, max_concurrent):
async def new_parallel_exps4(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], jobs : int):
    sem = asyncio.Semaphore(jobs)

    async def semaphore_task(data):
        async with sem:
            return await new_single_exp3(*data)

    tasks = [semaphore_task((df_results,) + data) for data in zip(conf_algs, conf_exps, [f'tmp{i}.txt' for i in range(len(conf_algs))])]
    results = await asyncio.gather(*tasks)

def process_task(df_results, data):
    # This function wraps the `run_process` to handle exceptions
    try:
        ret = new_single_exp2(*data)
        df_results.loc[len(df_results)] = ret
        return ret
    except Exception as e:
        return f"Process {data['command']} generated an exception: {e}"

def new_parallel_exps5(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], jobs : int):
    with Pool(processes=jobs) as pool:
        results = []

        for data in zip(conf_algs, conf_exps):
            result = pool.apply_async(process_task, (df_results, data,), callback=handle_output_callback)

        pool.close()
        pool.join()

def handle_output_callback(result):
    if isinstance(result, str) and result.startswith("Process"):
        print(result)
    else:
        pass

def new_multiple_exps(df_results : pd.DataFrame, conf_algs : list[alg_config], conf_exps : list[exp_config], tmp_filename : str, save_to_csv_as : Union[None, str] = None, timeout : Union[None, float] = None) -> None:

    visited_graphs = {}  # graph to (number of nodes, number of edges)
    for conf_alg, conf_exp in zip(conf_algs, conf_exps):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        times = run_single_exp(conf_alg, conf_exp, tmp_filename, timeout=timeout)

        print('Avg time: ', times[0])

        # print('Loading graph info...')
        n, m = get_graph_info(conf_exp.filename) if conf_exp.filename not in visited_graphs else visited_graphs[conf_exp.filename]
        visited_graphs[conf_exp.filename] = (n, m)
        # print('Done.')

        df_exp_name = 'None' if save_to_csv_as is None else os.path.splitext(os.path.basename(save_to_csv_as))[0]

        # Add the results to the dataframe
        df_results.loc[len(df_results)] = [timestamp] + conf_exp.to_list() + conf_alg.to_list() + times + [n, m] + [df_exp_name]

        if save_to_csv_as is not None:
            df_results.to_csv(save_to_csv_as, index=False)



def set_of_exps(exp_name : str, conf_algs : list[alg_config], conf_exps : list[exp_config], timeout : Union[None, float] = None) -> None:
    os.chdir("../build")

    result_filename = f'{exp_name}.csv'

    abs_result_filename = os.path.join('../data/experiments', result_filename)

    df_results = get_exp_file(abs_result_filename)

    new_multiple_exps(df_results, conf_algs, conf_exps, f'tmp_{exp_name}.txt', save_to_csv_as=abs_result_filename, timeout=timeout)


# %%
def main():
    os.chdir("../build")

    df = get_exp_file("../scripts/results.csv")

    conf_alg1 = alg_config(cutedges=1)
    conf_alg2 = alg_config(cutedges=2)
    conf_exp12 = exp_config(alg_goal="SSSP",
                        exp_type="time",
                        alg_name="BCF",
                        filename="../data/graphs/med_aug_bfct_10e6.txt",
                        source=0,
                        reps=2)

    # run_single_exp(conf_alg1, conf_exp1)
    # new_single_exp(df, conf_alg2, conf_exp12)
    # new_parallel_exps(df, [conf_alg2], [conf_exp2], 3)
    # new_parallel_exps2(df, [conf_alg1, conf_alg2], [conf_exp12, conf_exp12], 5)
    # new_parallel_exps3(df, [conf_alg1, conf_alg2], [conf_exp12, conf_exp12], 3)
    # asyncio.run(new_parallel_exps4(df, [conf_alg1, conf_alg2], [conf_exp12, conf_exp12], 3))
    # new_parallel_exps5(df, [conf_alg1, conf_alg2], [conf_exp12, conf_exp12], 3)
    # new_multiple_exps(df, [conf_alg1, conf_alg2], [conf_exp12, conf_exp12])


    conf_exp34 = exp_config(alg_goal="SSSP",
                        exp_type="time",
                        alg_name="BCF",
                        filename="../data/graphs/big_aug_bfct_10e7.txt",
                        source=0,
                        reps=1)


    new_single_exp(df, conf_alg2, conf_exp34)
    new_parallel_exps3(df, [conf_alg1, conf_alg2], [conf_exp34, conf_exp34], 3)

    df.to_csv("../scripts/results.csv", index=False)

    # Print the last three entries of the column "avg_time"
    print(df["avg_time"].tail(3))


# %%
if __name__ == "__main__":
    main()
