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

    for req in requests:
        # 1. Tokenize
        tokens = tokenize_prompt(req.input, tokenizer)

        # 2. KV block hashes
        block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)

        # 3. Longest-prefix scoring (inline)
        best_pod = 0
        best_score = -1

        for pod_id, pod_cache in POD_KV_CACHE.items():
            score = 0
            for key in block_keys:
                if key in pod_cache:
                    score += 1
                else:
                    break
            if score > best_score:
                best_score = score
                best_pod = pod_id

        # =========================
        # Cold-start fairness (Option 1)
        # =========================
        if best_score == 0:
            # round-robin based on request index
            best_pod = len(policy) % num_sims

        # 4. Route
        policy.append(best_pod)

        # 5. Update KV cache AFTER routing
        POD_KV_CACHE[best_pod].update(block_keys) # DONT CHANGE HERE!!!

    # Additional: enforce prefix dominance for pod 0 only if prefix matches > 3 and best_pod != 0
    prefix_length_threshold = 4
    for idx, req in enumerate(requests):
        if policy[idx] != 0:
            tokens = tokenize_prompt(req.input, tokenizer)
            block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)
            pod0_cache = POD_KV_CACHE[0]
            common_prefix = 0
            for key in block_keys:
                if key in pod0_cache:
                    common_prefix += 1
                else:
                    break
            if common_prefix >= prefix_length_threshold:
                # Override to pod 0 to enforce prefix dominance when prefix is long enough
                policy[idx] = 0
                POD_KV_CACHE[0].update(block_keys)

    # Additional: spillover traffic forcibly from pod 0 to pod 1 every 4th request (increase spill frequency)
    if num_sims > 1:
        for i in range(len(policy)):
            if policy[i] == 0 and (i + 1) % 4 == 0:
                tokens = tokenize_prompt(requests[i].input, tokenizer)
                block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)
                policy[i] = 1
                POD_KV_CACHE[1].update(block_keys)

    return policy

# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return prefix_aware_router