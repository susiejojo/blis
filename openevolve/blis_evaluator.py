from typing import List
import json
import csv
import os
import subprocess
from transformers import AutoTokenizer
from dataclasses import dataclass

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
    input_len: int             
    output_len: int            
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

    traces_filepath = f"traces_instance_{simulator_instance}.csv"
    write_requests_to_csv(requests, traces_filepath, instance_config["model"])
    model_name = instance_config["model"].split("/")[1].lower()

    blis_args = ["run"]
    for k,v, in instance_config.items():
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

if __name__=="__main__":
    requests = [
        InferenceRequest(arrival_time=0, input_len=4, output_len=2, input="hi how are you", output="All good"),
        InferenceRequest(arrival_time=0.1, input_len=2, output_len=3, input="hello yes", output="In the day"),
    ] 

    mean_e2e = call_blis(1, requests)
    print("Mean E2E:", mean_e2e)
    