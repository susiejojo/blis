"""
Prefix-aware router (LLMD-style).

- Global pod KV cache
- Deterministic KV block hashing (struct.pack)
- Longest-prefix scoring
- Offline, sequential routing (OpenEvolve / BLIS friendly)
"""

import hashlib
import struct
from typing import List, Dict, Any

from transformers import AutoTokenizer


# =========================
# Global KV cache per pod
# =========================

POD_KV_CACHE: Dict[int, set] = {}


def tokenize_prompt(prompt: str, tokenizer) -> List[int]:
    """Tokenize prompt into token IDs."""
    return tokenizer.encode(prompt, add_special_tokens=False)


def tokens_to_kv_block_keys(
    tokens: List[int],
    kv_block_size: int = 16,
) -> List[str]:
    """Convert token IDs into KV block hashes (uint32 packed)."""
    block_keys = []

    for i in range(0, len(tokens), kv_block_size):
        chunk = tokens[i : i + kv_block_size]
        h = hashlib.sha256()
        for t in chunk:
            h.update(struct.pack("I", t))  # uint32
        block_keys.append(h.hexdigest())

    return block_keys

# EVOLVE-BLOCK-START
def prefix_aware_router(
    requests: List[Any],
    num_sims: int,
    model_name: str = "codellama/CodeLlama-34b-Instruct-hf", # no need to change this - we use default
    kv_block_size: int = 16, # no need to change this - we use default
) -> List[int]:
    """
    Prefix-aware router.

    Args:
        requests: list of objects with `.input`
        num_sims: number of pods / simulators

    Returns:
        policy: chosen pod per request
    """
    global POD_KV_CACHE

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize pod KV cache
    if not POD_KV_CACHE or len(POD_KV_CACHE) != num_sims:
        POD_KV_CACHE = {pod_id: set() for pod_id in range(num_sims)}

    policy = []
    pod_request_count = [0] * num_sims  # Track requests per pod

    for req in requests:
        # 1. Tokenize
        tokens = tokenize_prompt(req.input, tokenizer)

        # 2. KV block hashes
        block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)

        # 3. Longest-prefix scoring with weighted blocks
        best_pods = []
        best_score = -1

        for pod_id, pod_cache in POD_KV_CACHE.items():
            score = 0.0
            # Weight early blocks more heavily for better cache locality
            for idx, key in enumerate(block_keys):
                if key in pod_cache:
                    # Exponential decay: first block worth 1.0, second 0.9, etc.
                    score += 0.9 ** idx
                else:
                    break
            if score > best_score:
                best_score = score
                best_pods = [pod_id]
            elif score == best_score:
                best_pods.append(pod_id)
        
        # Tie-breaking: choose least loaded pod among ties
        if len(best_pods) > 1:
            # Use the pod_request_count we're already tracking
            best_pod = min(best_pods, key=lambda p: pod_request_count[p])
        else:
            best_pod = best_pods[0]

        # =========================
        # Cold-start fairness (Option 1)
        # =========================
        if best_score == 0:
            # round-robin based on request index
            best_pod = len(policy) % num_sims

        # 4. Route
        policy.append(best_pod)
        pod_request_count[best_pod] += 1  # Actually use the counter

        # 5. Update KV cache with size management
        # Add new blocks but limit cache growth
        POD_KV_CACHE[best_pod].update(block_keys)
        
        # Simple cache eviction if too large (keep most recent 1000 blocks)
        max_cache_size = 1000
        if len(POD_KV_CACHE[best_pod]) > max_cache_size:
            # Keep only the most recent blocks by converting to list and taking last N
            POD_KV_CACHE[best_pod] = set(list(POD_KV_CACHE[best_pod])[-max_cache_size:])

    return policy

# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return prefix_aware_router