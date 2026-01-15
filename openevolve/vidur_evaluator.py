from typing import List
import json
import csv
import os
import sys
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd
from dataclasses import dataclass

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
    write_requests_to_csv(requests, traces_filepath, instance_config["model"])
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

if __name__=="__main__":
    requests = [
        InferenceRequest(arrival_time=0, input="hi how are you", output="All good"),
        InferenceRequest(arrival_time=0.1, input="hello yes", output="In the day"),
    ] 

    mean_e2e = call_vidur(1, requests)
    print("Mean E2E:", mean_e2e)
    