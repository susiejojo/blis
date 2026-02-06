from collections import defaultdict
import pickle
from typing import List
import json
import csv
import os
import subprocess
from transformers import AutoTokenizer
from dataclasses import dataclass
import importlib.util
from openevolve.evaluation_result import EvaluationResult
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import csv
import random
import time
import pickle 
from request_types import InferenceRequest
from generate_req import create_repeated_req_per_simulator, create_repeated_req_per_simulator_w_suffix, create_repeated_req, create_high_repeated_decode_heavy


random.seed(42)

BLIS_BINARY_PATH = "./simulation_worker"
INSTANCE_CONFIG_PATH = "routing_instance_config.json"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIGS_TO_VIDUR_ARGS_MAPPING = {
    "model": "replica_config_model_name",
    "max-num-batched_tokens": "vllm_scheduler_config_max_tokens_in_batch",
    "max-num-seqs": "vllm_scheduler_config_batch_size_cap",
    "total-kv-blocks": "vllm_scheduler_config_num_blocks",
    "GPU": "replica_config_device",
    "tp": "replica_config_tensor_parallel_size",
}

CONFIGS_TO_BLIS_ARGS_MAPPING = {
    "model": "model",
    "max-num-batched_tokens": "max-num-scheduled-tokens",
    "max-num-seqs": "max-num-running-reqs",
    "total-kv-blocks": "total-kv-blocks",
    "GPU": "hardware",
    "tp": "tp",
    "vllm-version": "vllm-version"
}

def write_requests_to_csv(requests, output_path, model_id="gpt2"):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow([
            "arrived_at", 
            "num_prefill_tokens", 
            "num_decode_tokens", 
            "prefill_tokens", 
            "decode_tokens"
        ])

        for req in requests:
            # Tokenize strings into lists of integers
            prefill_tokens = tokenizer.encode(req.input, add_special_tokens=False)
            decode_tokens = tokenizer.encode(req.output, add_special_tokens=False)

            # Use json.dumps to ensure the list is written as "[1, 2, 3]" 
            writer.writerow([
                req.arrival_time,
                len(prefill_tokens),
                len(decode_tokens),
                json.dumps(prefill_tokens),
                json.dumps(decode_tokens)
            ])

def run_go_binary(arguments, go_binary_path):
    """
    Run BLIS's Go binary and return metrics (mean e2e) per instance.
    
    :param arguments: BLIS args
    :param go_binary_path: Path to BLIS Go binary
    """
    result = subprocess.run(
        [go_binary_path] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )

    if result.stderr:
        print(
            f"Go binary error output:\n{result.stderr}")
    json_start = result.stdout.find('{')
    json_data = result.stdout[json_start:]

    # Parse into a dictionary, and extract mean E2E
    metrics = json.loads(json_data)
    mean_e2e = metrics["e2e_mean_ms"]
    return mean_e2e

def call_blis(
    simulator_instance: int, # instance ID
    requests: List[InferenceRequest],
):
    """
    Preprocesses requests into CSV consumable by BLIS,
    then parses BLIS output to get mean E2E in milliseconds.
    It returns the mean E2E value if the BLIS run is successful,
    otherwise it returns Inf.
    
    :param simulator_instance: BLIS Instance ID
    :type simulator_instance: int
    :param requests: List of requests to be sent to this BLIS instance
    :type requests: List[InferenceRequest]
    """
    with open(INSTANCE_CONFIG_PATH, "r") as f:
        instance_config = json.load(f)

    traces_filepath = f"blis_traces_instance_{simulator_instance}.csv"
    write_requests_to_csv(requests, traces_filepath, instance_config["model"])
    model_name = instance_config["model"].split("/")[1].lower()

    blis_args = ["run"]
    for k,v in instance_config.items():
        arg = CONFIGS_TO_BLIS_ARGS_MAPPING[k]
        blis_args.extend([f"--{arg}", str(v)])
    extra_args = {
        "workload": "traces",
        "workload-traces-filepath": traces_filepath,
        "horizon": "922337203685477580", # Golang int64 max value
        "log": "fatal",
        "model-config-folder": f"model_configs/{model_name}",
        "hardware-config": "hardware_config.json"
    }
    for key in extra_args:
        blis_args.extend([f"--{key}", str(extra_args[key])])

    try:
        instance_e2e = run_go_binary(blis_args, BLIS_BINARY_PATH)
        return instance_e2e
    except Exception as e:
        print(f"ERROR: {e}")
        return float("Inf")

def write_requests_to_csv_vidur(requests, output_path, model_id="gpt2"):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow([
            "arrived_at", 
            "num_prefill_tokens", 
            "num_decode_tokens"
        ])

        for req in requests:
            # Tokenize strings into lists of integers
            prefill_tokens = tokenizer.encode(req.input, add_special_tokens=False)
            decode_tokens = tokenizer.encode(req.output, add_special_tokens=False)

            # Use json.dumps to ensure the list is written as "[1, 2, 3]" 
            writer.writerow([
                req.arrival_time,
                len(prefill_tokens),
                len(decode_tokens),
            ])

def run_vidur(arguments, vidur_output_dir):
    """
    Run vidur and return metrics (mean e2e) per instance.
    
    :param arguments: vidur args
    :param go_binary_path: Path to vidur Go binary
    """
    cmd = ["python", "-m", "vidur.main"]
    result = subprocess.run(
        cmd + arguments,
        capture_output=True,
        cwd=os.path.join(os.getcwd(), "vidur"),
        text=True,
        check=True,
        encoding='utf-8'
    )

    if result.stderr:
        print(
            f"Vidur run error output:\n{result.stderr}")
        
    # Find results directory
    results_folder = Path(vidur_output_dir)
    dirs = sorted([d for d in results_folder.iterdir() if d.is_dir()]) # latest run
    results_df = pd.read_csv(f"{dirs[-1]}/request_metrics.csv")

    # Extract mean E2E from individual requests
    mean_e2e = results_df["request_e2e_time"].mean() * 1e3 # sec to millisec
    return mean_e2e

def call_vidur(
    simulator_instance: int, # instance ID
    requests: List[InferenceRequest],
):
    """
    Preprocesses requests into CSV consumable by vidur,
    then parses vidur output to get mean E2E in milliseconds.
    It returns the mean E2E value if the vidur run is successful,
    otherwise it returns Inf.
    
    :param simulator_instance: vidur Instance ID
    :type simulator_instance: int
    :param requests: List of requests to be sent to this vidur instance
    :type requests: List[InferenceRequest]
    """
    with open(INSTANCE_CONFIG_PATH, "r") as f:
        instance_config = json.load(f)

    current_dir = os.getcwd() # Assumes you are in openevolve
    traces_filepath = f"{current_dir}/vidur_traces_instance_{simulator_instance}.csv"
    write_requests_to_csv_vidur(requests, traces_filepath, instance_config["model"])
    vidur_output_dir = f"{current_dir}/vidur_results_instance_{simulator_instance}"

    vidur_args = []
    for k,v, in instance_config.items():
        if k in CONFIGS_TO_VIDUR_ARGS_MAPPING:
            arg = CONFIGS_TO_VIDUR_ARGS_MAPPING[k]
            if k == "GPU": # Vidur accepts GPU names like: h100, not H100
                v = v.lower()
            vidur_args.extend([f"--{arg}", str(v)])
    network_device = "h100_dgx" if instance_config["GPU"] == "H100" else "a100_dgx"
    extra_args = {
        "length_generator_config_type": "trace",
        "trace_request_length_generator_config_trace_file": traces_filepath,
        "time_limit": "922337203685477580", # Golang int64 max value
        "metrics_config_output_dir": vidur_output_dir,
        "replica_config_network_device": network_device,
        "replica_scheduler_config_type": "vllm",
        "synthetic_request_generator_config_num_requests": len(requests),
        "cluster_config_num_replicas": 1,
    }
    for key in extra_args:
        vidur_args.extend([f"--{key}", str(extra_args[key])])

    vidur_args.append("--no-metrics_config_store_plots") # Don't save plots
    try:
        instance_e2e = run_vidur(vidur_args, vidur_output_dir)
        return instance_e2e
    except Exception as e:
        print(f"ERROR: {e}")
        return float("Inf")
   
def random_word(min_len=3, max_len=10):
    import string
    letters = string.ascii_lowercase
    return "".join(random.choices(letters, k=random.randint(min_len, max_len)))


# -----------------------------
# OpenEvolve evaluator
# -----------------------------
# requests = generate_requests(n=4800)
def create_repeated_req_per_set(num_sims=4, num_reqs=200, content_len=5000):
    """
    Create a request stream where:
    - Most requests are unique
    - Every (num_sims + 1)-th request is an exact duplicate of the previous one
      (used to test prefix caching + routing stickiness)
    """

    random.seed(42)
    shared_body = random_word() + " "
    output_word = random_word()

    requests = []
    t = 0.0
    counter = 1

    while len(requests) < num_reqs:
        # Create exact duplicates every (num_sims + 1)-th request
        # e.g. for num_sims=4 → requests 5, 9, 13, ...
        if counter % (num_sims + 1) == 0:
            prompt = f"{counter - 1} " + (shared_body * content_len)
        else:
            prompt = f"{counter} " + (shared_body * content_len)

        requests.append(
            InferenceRequest(
                arrival_time=t,
                input=prompt,
                output=output_word,
            )
        )

        t += 1
        counter += 1

    return requests

NUM_SIMS = 4
# REQUESTS = create_repeated_req_per_simulator(num_reqs=100, content_len=2000, num_sims=NUM_SIMS)
# REQUESTS = create_repeated_req_per_simulator_w_suffix(num_reqs=100, content_len=2000, num_sims=NUM_SIMS)
# REQUESTS = create_repeated_req_per_simulator_w_suffix(num_reqs=212, content_len=2050, num_sims=NUM_SIMS, reqs_per_sec=10)
# REQUESTS = create_repeated_req(num_reqs=212, content_len=2050, num_sims=NUM_SIMS, reqs_per_sec=10)
REQUESTS = create_high_repeated_decode_heavy(num_reqs=300, content_len=200, num_sims=NUM_SIMS, reqs_per_sec=15) # random is better than prefix 
print("reqs", len(REQUESTS))
def evaluate(program_path):
    try:
        # load evolved router
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        router = program.run_search()

        print("----called here")

        policy = router(REQUESTS, num_sims=NUM_SIMS)
        print(policy)

        # safety guard
        if not isinstance(policy, list) or len(policy) != len(REQUESTS):
            # print(0/0)
            print("ERRR")
            return EvaluationResult(
                metrics={"score": -1e9},
                artifacts={"error": "Invalid routing policy"}
            )

        # split requests
        buckets = [[] for _ in range(NUM_SIMS)]
        for req, sim_id in zip(REQUESTS, policy):
            buckets[int(sim_id)].append(req)

        # Run simulators
        latencies = []
        counts = []

        for sim_id in range(NUM_SIMS):
            reqs = buckets[sim_id]
            cnt = len(reqs)
            lat = call_blis(sim_id, reqs) if cnt > 0 else 0.0
            print("lat: ", lat, " cnt: ", cnt)
            # lat = call_blis_blackbox(sim_id, reqs) if cnt > 0 else 0.0
            # lat = call_vidur(sim_id, reqs) if cnt > 0 else 0.0
            latencies.append(lat)
            counts.append(cnt)

        total_reqs = sum(counts)
        avg_latency = (
            sum(lat * cnt for lat, cnt in zip(latencies, counts)) / total_reqs
            if total_reqs > 0 else 0.0
        )
        print("avg:" , avg_latency)


        # OpenEvolve maximizes score → minimize latency
        score = -avg_latency

        return EvaluationResult(
            metrics={"combined_score": score},
            artifacts={
                "avg_latency": avg_latency,
                # "sim0_requests": req0,
                # "sim1_requests": req1,
            }
        )

    except Exception as e:
        print("---------EVALUATION ERROR:", e)
        return EvaluationResult(
            metrics={"score": -1e9},
            artifacts={"error": str(e)}
        )

def evaluate_stage1(program_path):
    return evaluate(program_path)

def evaluate_stage2(program_path):
    return evaluate(program_path)

