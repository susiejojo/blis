import json
import pickle
from dataclasses import dataclass
from typing import List


@dataclass
class InferenceRequest:
    arrival_time: float
    input: str
    output: str


def build_requests(json_path: str) -> List[InferenceRequest]:
    with open(json_path, "r") as f:
        data = json.load(f)

    requests = []
    t0 = data[0]["start_time"] if data else 0.0

    for rec in data:
        # parse request JSON string
        req_payload = json.loads(rec["request"])
        prompt = req_payload.get("prompt", "")

        # parse response JSON string
        resp_payload = json.loads(rec["response"])
        choices = resp_payload.get("choices", [])
        output = choices[0].get("text", "") if choices else ""

        requests.append(
            InferenceRequest(
                arrival_time=rec.get("start_time", 0.0) - t0,
                input=prompt,
                output=output,
            )
        )

    return requests


def dump_pickle(reqs: List[InferenceRequest], out_path: str):
    with open(out_path, "wb") as f:
        pickle.dump(reqs, f)


if __name__ == "__main__":
    json_file = "inferenceperf-reqs.json"
    pickle_file = "inference_requests.pkl"

    reqs = build_requests(json_file)
    print(reqs[0:2])
    dump_pickle(reqs, pickle_file)

    print(f"Saved {len(reqs)} requests → {pickle_file}")
