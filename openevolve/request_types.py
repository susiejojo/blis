# request_types.py
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    arrival_time: float
    input: str
    output: str