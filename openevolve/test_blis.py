from collections import defaultdict
from pathlib import Path
import pickle
from typing import List
import json
import csv
import os
import subprocess
import pandas as pd
from transformers import AutoTokenizer
from dataclasses import dataclass

from dataclasses import dataclass
import csv
import random

from generate_req import create_repeated_req_per_simulator, create_repeated_req_per_simulator_w_suffix, create_repeated_req, create_high_repeated_decode_heavy

from initial_llm_router import prefix_aware_router

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

@dataclass
class InferenceRequest:
    arrival_time: float    # in seconds             
    input: str  
    output: str

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
    # print(metrics)
    mean_e2e = metrics["e2e_mean_ms"]
    p90 = metrics["e2e_p90_ms"]

    return mean_e2e, p90

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
        instance_e2e, p90 = run_go_binary(blis_args, BLIS_BINARY_PATH)
        return instance_e2e, p90
    except Exception as e:
        print(f"ERROR: {e}")
        return float("Inf"),float("Inf")
    
def call_blis_blackbox(
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
        "alpha-coeffs": "10326.268121600151,0.0,0.0",
        "beta-coeffs": "6472.007648778123,19.79721387814611,304.4021040302657",
        # "model-config-folder": f"model_configs/{model_name}",
        # "hardware-config": "hardware_config.json"
    }
    for key in extra_args:
        blis_args.extend([f"--{key}", str(extra_args[key])])

    try:

        instance_e2e = run_go_binary(blis_args, BLIS_BINARY_PATH)
        return instance_e2e
    except Exception as e:
        print(f"ERROR: {e}")
        return float("Inf")

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
    
# generate_requests_once.py
from dataclasses import dataclass
import pickle
import random

random.seed(42)


def random_word(min_len=2, max_len=3):
    import string
    letters = string.ascii_lowercase
    return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

def random_words(n):
    return " ".join(random_word() for _ in range(n))

import hashlib

def extract_prefix(request, n_words=20):
    return " ".join(request.input.split()[:n_words])
    
def router_always0(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # initial dumb policy (random)
        policy.append(1)
    return policy

def router_rand(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # initial dumb policy (random)
        policy.append(random.randint(0, num_sims - 1))
    return policy

def router_prefix(requests, num_sims=2, n_prefix_words=50):
    """
    Prefix-hash router:
    - Extracts the first `n_prefix_words` from each request
    - Hashes that prefix
    - Deterministically maps the request to a simulator
    """

    policy = []

    for req in requests:
        # Extract routing key (string prefix)
        prefix = extract_prefix(req, n_prefix_words)

        # Stable hash → simulator id
        h = int(hashlib.sha256(prefix.encode()).hexdigest(), 16)
        sim_id = h % num_sims

        policy.append(sim_id)

    return policy

def router_rrb(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    current_sim_idx = 0 # Initialize a counter for round-robin routing
    for req in requests:
        # The previous hash-based routing failed due to 'unhashable type: InferenceRequest'.
        # Implementing a simple round-robin policy to ensure even load distribution across simulators.
        # This strategy provides a basic form of load balancing and avoids the hashability issue.
        sim_id = current_sim_idx % num_sims
        policy.append(sim_id)
        current_sim_idx += 1 # Move to the next simulator for the next request
    return policy

def router_prefix_lb(requests, num_sims=2, n_prefix_tokens=20):
    """
    Prefix-aware load-balancing router.

    - Requests are grouped by a token-level prefix.
    - Each prefix is "sticky" to one simulator.
    - Stickiness may be broken if that simulator becomes overloaded.
    """

    policy = []
    sim_load = defaultdict(int)   # number of requests per simulator
    prefix_to_sim = {}            # sticky mapping: prefix -> simulator

    for req in requests:
        # Routing key: first n_prefix_tokens of the request
        prefix = extract_prefix(req, n_prefix_tokens)

        if prefix in prefix_to_sim:
            # Reuse simulator previously assigned to this prefix
            sim_id = prefix_to_sim[prefix]

            # Optional load-shedding: break stickiness if overloaded
            min_load = min(sim_load.values()) if sim_load else 0
            if sim_load[sim_id] > min_load + 5:
                sim_id = min(range(num_sims), key=lambda i: sim_load[i])
                prefix_to_sim[prefix] = sim_id
        else:
            # First time seeing this prefix:
            #   - hash prefix to choose a starting point
            #   - select least-loaded simulator (consistent hashing style)
            h = int(hashlib.sha256(prefix.encode()).hexdigest(), 16)
            start = h % num_sims
            sim_id = min(
                ((start + i) % num_sims for i in range(num_sims)),
                key=lambda i: sim_load[i],
            )
            prefix_to_sim[prefix] = sim_id

        policy.append(sim_id)
        sim_load[sim_id] += 1

    return policy

def openevolve_vidur_router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # Route requests based on the prefix of their input to leverage locality.
        # A block is often 16 tokens.
        prefix_len = 16
        
        # Safely get the input prefix. Assume req.input is a sequence of tokens.
        # Convert to a tuple to ensure hashability and consistent hashing.
        if hasattr(req, 'input') and req.input is not None:
            input_prefix = tuple(req.input[:prefix_len])
        else:
            # If no input, or input is None, use an empty tuple for consistent hashing.
            # This ensures all requests without input are routed to the same simulator.
            input_prefix = ()

        # Hash the prefix to determine the simulator ID
        simulator_id = hash(input_prefix) % num_sims
        policy.append(simulator_id)
    return policy

def router_openevolve_blis(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    This improved router balances load while preserving request locality based on input prefixes.
    """
    policy = []
    
    # Store mappings from request prefixes to simulator IDs to preserve locality.
    # This allows requests with shared prefixes to benefit from being routed to the same simulator.
    prefix_to_sim_map = {}
    
    # Track the current number of requests assigned to each simulator.
    # This helps with load balancing when assigning new prefixes.
    sim_loads = [0] * num_sims 

    for req in requests:
        # Extract the prefix for locality. The problem hints at "first few blocks"
        # and "a block is often 16 tokens". We'll use the first 16 elements/characters.
        prefix_length = 16
        # Slice the input to get the prefix segment.
        prefix_segment = req.input[:min(len(req.input), prefix_length)]
        # Convert the prefix segment to a hashable type (e.g., tuple) to use as a dictionary key.
        # This handles both string inputs (tuple of chars) and list of token inputs (tuple of tokens).
        prefix = tuple(prefix_segment) 

        sim_id = -1
        if prefix in prefix_to_sim_map:
            # If this prefix has been seen before, route to the same simulator
            # to preserve locality and leverage potential simulator state/cache.
            sim_id = prefix_to_sim_map[prefix]
        else:
            # If it's a new prefix, assign it to the simulator with the minimum current load.
            # This helps balance the load across simulators for unseen prefixes,
            # ensuring no single simulator becomes a bottleneck prematurely.
            min_load = float('inf')
            min_load_sim_id = -1
            for i in range(num_sims):
                if sim_loads[i] < min_load:
                    min_load = sim_loads[i]
                    min_load_sim_id = i
            
            sim_id = min_load_sim_id
            prefix_to_sim_map[prefix] = sim_id # Store the mapping for future requests with this prefix
        
        policy.append(sim_id)
        sim_loads[sim_id] += 1 # Increment the load for the assigned simulator
        
    return policy

def test_routers_gpt(num_sims=4):
    routers = {
        "random": router_rand,
        "router0": router_always0,
        # # "round_robin": router_rrb,
        
        # # # "prefix_lb": router_prefix_lb,
        # # # "openevolve_vidur_router": openevolve_vidur_router,
        # # # "prefix_hash": router_prefix,
        # "router_openevolve_blis": router_openevolve_blis,
        "router_llmd": prefix_aware_router,
    }

    # requests = create_repeated_req_per_simulator_w_suffix(num_reqs=50, content_len=2000, num_sims=num_sims, reqs_per_sec=1)
    # requests = create_repeated_req(num_reqs=93, content_len=1700, num_sims=num_sims, reqs_per_sec=20)
    # requests = create_high_repeated_decode_heavy(num_reqs=120, content_len=2, num_sims=num_sims, reqs_per_sec=10) # decode heavy but prefix 80% - so prefix aware does bad job

    num_sims = 4
    requests = create_high_repeated_decode_heavy(num_reqs=120, content_len=200, num_sims=num_sims, reqs_per_sec=15)
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-Instruct-hf")
    # for req in requests:
    #     print(req.input)
    #     # Tokenize strings into lists of integers
    #     prefill_tokens = tokenizer.encode(req.input, add_special_tokens=False)
    #     print(prefill_tokens)
    # print(requests)

    for name, router_fn in routers.items():

        # Run router
        if name == "prefix_lb":
            policy = router_fn(requests, num_sims, 20)
        else:
            policy = router_fn(requests, num_sims)

        print(policy)

        # Split requests
        buckets = [[] for _ in range(num_sims)]
        for req, sim_id in zip(requests, policy):
            buckets[int(sim_id)].append(req)

        # Run simulators
        latencies = []
        latencies_p90 = []
        counts = []

        for sim_id in range(num_sims):
            reqs = buckets[sim_id]
            cnt = len(reqs)
            lat, p90 = call_blis(sim_id, reqs) if cnt > 0 else (0.0, 0.0)
            # print(lat,p90)
            # lat = call_blis_blackbox(sim_id, reqs) if cnt > 0 else 0.0
            # lat = call_vidur(sim_id, reqs) if cnt > 0 else 0.0
            latencies.append(lat)
            latencies_p90.append(p90)
            counts.append(cnt)

        # print("p90", sum(latencies_p90)/sum(counts))

        total_reqs = sum(counts)
        avg_lat = (
            sum(lat * cnt for lat, cnt in zip(latencies, counts)) / total_reqs
            if total_reqs > 0 else 0.0
        )

        print(f"\n=== Router: {name} ===")
        for i in range(num_sims):
            print(f"sim{i}: {counts[i]} reqs, lat{i} = {latencies[i]:.3f}")
        print(f"avg latency: {avg_lat:.3f}")

if __name__=="__main__":


    print("\n\n\n")

    # test_routers(num_sims=8)
    test_routers_gpt(num_sims=2)


