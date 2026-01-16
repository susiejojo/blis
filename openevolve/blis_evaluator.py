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

from dataclasses import dataclass
import csv
import random
import time
random.seed(42)

BLIS_BINARY_PATH = "./simulation_worker"
INSTANCE_CONFIG_PATH = "routing_instance_config.json"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def generate_requests(n=10, start_time=0.0):
    """
    - Read prompts from promptsblisopenevolve.txt (separated by blank lines)
    - Generate arrival times (~5 req/sec)
    - Random output lengths [16, 256]
    """

    def read_prompts(path):
        with open(path, "r") as f:
            text = f.read()

        return [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
    print("--now here")
    prompts = read_prompts("data/promptsblisopenevolve.txt")
    print("--now here2")
    assert prompts, "No prompts found!"

    requests = []
    current_time = start_time

    for i in range(n):
        # inter-arrival time (Poisson process)
        current_time += random.expovariate(5)  # avg 5 req/sec

        # pick a prompt (cycle if n > num_prompts)
        prompt = prompts[i % len(prompts)]

        output_len = random.randint(16, 64)

        inp = prompt
        out = "x " * output_len  # simple synthetic output

        req = InferenceRequest(
            arrival_time=round(current_time, 6),
            input=inp,
            output=out.strip(),
        )
        requests.append(req)

    return requests

def random_word(min_len=3, max_len=10):
    import string
    letters = string.ascii_lowercase
    return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

def random_words(n):
    return " ".join(random_word() for _ in range(n))

# def generate_requests_dummy(n=100):
#     """
#     - Prefix length: 23–128 WORDS
#     - Prefix words: completely random
#     - High prefix reuse to show routing benefit
#     """
#     # Create a small pool of reusable random prefixes
#     print("creating random requests of ", n)
#     num_prefixes = 5
#     prefix_pool = []

#     for _ in range(num_prefixes):
#         prefix_len = random.randint(23, 128)
#         prefix_pool.append(random_words(prefix_len))

#     requests = []
#     t = 0.0

#     for _ in range(n):
#         prefix = random.choice(prefix_pool)

#         # Long, variable suffix (noise / continuation)
#         suffix_words = random.randint(50, 200)
#         suffix = random_words(suffix_words)

#         req = InferenceRequest(
#             arrival_time=t,
#             input=prefix + " " + suffix,
#             output=""
#         )

#         requests.append(req)
#         t += random.expovariate(8)  # bursty arrivals

#     return requests

# def generate_requests_dummy(n=300):
    print("creating random requests of ", n)

    num_prefixes = 4
    prefix_pool = [
        random_words(random.randint(100, 200))
        for _ in range(num_prefixes)
    ]

    requests = []
    t = 0.0

    burst_size = 25
    for prefix in prefix_pool:
        for _ in range(burst_size):
            suffix = random_words(random.randint(10, 30))

            requests.append(
                InferenceRequest(
                    arrival_time=t,
                    input=prefix + " " + suffix,
                    output=""
                )
            )

            t += random.expovariate(20)  # tight burst

        t += 2.0  # gap between prefix bursts

    return requests[:n]

def generate_requests_dummy():
    with open("requests.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------
# OpenEvolve evaluator
# -----------------------------
# requests = generate_requests(n=4800)
REQUESTS = generate_requests_dummy()

def evaluate(program_path):
    try:
        # load evolved router
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        router = program.run_search()

        print("----called here")
        
        # print(requests)
        # for req
        policy = router(REQUESTS, num_sims=2)

        # safety guard
        if not isinstance(policy, list) or len(policy) != len(REQUESTS):
            # print(0/0)
            return EvaluationResult(
                metrics={"score": -1e9},
                artifacts={"error": "Invalid routing policy"}
            )

        # split requests
        buckets = [[], []]
        for req, sim_id in zip(REQUESTS, policy):
            buckets[int(sim_id)].append(req)

        # run simulators
        # lat0, req0 = call_blis(0,buckets[0]), len(buckets[0])
        # lat1, req1 = call_blis(1,buckets[1]), len(buckets[1])
       
        # total_latency = lat0*req0 + lat1*req1
        # avg_latency = total_latency / (req1+req0) # len(requests)
        
        lat0, req0 = call_blis(0, buckets[0]), len(buckets[0])
        lat1,req1 = call_blis(1, buckets[1]), len(buckets[1])

        avg_latency = (lat0*req0 + lat1*req1) / (req0 + req1)

        # OpenEvolve maximizes score → minimize latency
        score = -avg_latency

        return EvaluationResult(
            metrics={"combined_score": score},
            artifacts={
                "avg_latency": avg_latency,
                "sim0_requests": req0,
                "sim1_requests": req1,
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

import hashlib

def extract_prefix(request, n_words=20):
    return " ".join(request.input.split()[:n_words])
    
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
    policy = []
    for req in requests:
        prefix = extract_prefix(req, n_prefix_words)
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
    Prefix-aware router with soft load balancing.
    
    - Same prefix → same preferred simulator
    - If overloaded, spill to next simulator
    - Uses only num_sims (no simulator objects)
    """
    policy = []

    # Track simulated load per simulator
    sim_load = defaultdict(int)

    # Cache prefix → simulator mapping for stickiness
    prefix_to_sim = {}

    for req in requests:
        prefix = extract_prefix(req, n_prefix_tokens)

        # If prefix already assigned, use it
        if prefix in prefix_to_sim:
            sim_id = prefix_to_sim[prefix]
        else:
            # Deterministic starting point from prefix hash
            h = int(hashlib.sha256(str(prefix).encode()).hexdigest(), 16)
            start = h % num_sims

            # Soft load balancing: probe simulators in round-robin order
            candidates = [(start + i) % num_sims for i in range(num_sims)]
            sim_id = min(candidates, key=lambda i: sim_load[i])

            prefix_to_sim[prefix] = sim_id

        policy.append(sim_id)
        sim_load[sim_id] += 1

    return policy

if __name__=="__main__":
    routers = {
        "random": router_rand,
        "round_robin": router_rrb,
        "prefix_hash": router_prefix,
        "prefix_lb": router_prefix_lb,
    }

    requests = generate_requests_dummy()
    num_sims = 2

    for name, router_fn in routers.items():

        # run router
        if name == "prefix_lb":
            policy = router_fn(requests, num_sims, 20)
        else:
            policy = router_fn(requests, num_sims)

        # split requests
        buckets = [[] for _ in range(num_sims)]
        for req, sim_id in zip(requests, policy):
            buckets[int(sim_id)].append(req)

        # run simulators
        lat0, req0 = call_blis(0, buckets[0]), len(buckets[0])
        lat1,req1 = call_blis(1, buckets[1]), len(buckets[1])

        avg_lat = (lat0*req0 + lat1*req1) / (req0 + req1)

        print(f"\n=== Router: {name} ===")
        print(f"sim0: {len(buckets[0])} reqs, lat0 = {lat0:.3f}")
        print(f"sim1: {len(buckets[1])} reqs, lat1 = {lat1:.3f}")
        print(f"avg latency: {avg_lat:.3f}")


    # requests = [
    #     InferenceRequest(arrival_time=0, input="hi how are you", output="All good"),
    #     InferenceRequest(arrival_time=0.1, input="hello yes", output="In the day"),
    # ] 

    # requests = generate_requests_dummy() # generate_requests(n=4800)
    # policy = router_prefix_lb(requests, 2, 20)

    # # split requests
    # buckets = [[], []]
    # for req, sim_id in zip(requests, policy):
    #     buckets[int(sim_id)].append(req)

    # # run simulators
    # lat0 = call_blis(0,buckets[0])
    # lat1 = call_blis(1,buckets[1])
    # print("lat0: ", lat0 , "for req: ", len(buckets[0]))
    # print("lat1: ", lat1 , "for req: ", len(buckets[1]))

    
    # total_latency = lat0 + lat1
    # avg_latency = total_latency / 2

    # # mean_e2e = call_blis(1, requests)
    # print("Mean E2E:", avg_latency)