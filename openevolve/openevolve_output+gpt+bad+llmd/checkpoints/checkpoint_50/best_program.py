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

        # 3. Longest-prefix scoring (inline) with tie break on pod_id parity and threshold for acceptance
        best_pod = -1
        best_score = -1
        tie_candidates = []

        for pod_id, pod_cache in POD_KV_CACHE.items():
            score = 0
            for key in block_keys:
                if key in pod_cache:
                    score += 1
                else:
                    break
            if score >= 2:
                if score > best_score:
                    best_score = score
                    tie_candidates = [pod_id]
                elif score == best_score:
                    tie_candidates.append(pod_id)

        if best_score < 2:
            # choose pod with smallest cache size to force spillover
            sizes = {pid: len(cache) for pid, cache in POD_KV_CACHE.items()}
            min_size = min(sizes.values())
            candidates = [pid for pid, size in sizes.items() if size == min_size]
            best_pod = candidates[len(policy) % len(candidates)]
        else:
            # break tie by picking even pod_id if exists else odd pod_id
            even_pods = [pid for pid in tie_candidates if pid % 2 == 0]
            if even_pods:
                best_pod = even_pods[len(policy) % len(even_pods)]
            else:
                best_pod = tie_candidates[len(policy) % len(tie_candidates)]

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

        # Additional: if pod 0 is heavily matched but best_pod is different and score close, spill traffic to pod 0 with threshold 3
        pod0_cache = POD_KV_CACHE[0]
        common_prefix_pod0 = 0
        prefix_threshold = 4  # raise threshold to require longer prefix for spillover
        for key in block_keys:
            if key in pod0_cache:
                common_prefix_pod0 += 1
            else:
                break
        if best_pod != 0 and common_prefix_pod0 >= prefix_threshold and (best_score - common_prefix_pod0) <= 0:
            # Spill to pod 0 only if it strictly ties or beats best prefix (tighter spill condition)
            best_pod = 0
            policy[-1] = best_pod
            POD_KV_CACHE[best_pod].update(block_keys)

        # Additional: force spillover from pod 0 to pod 1 every 4th request if pod 0 is dominant and request index mod 4 == 3
        if best_pod == 0 and num_sims > 1 and (len(policy) - 1) % 4 == 3:
            alt_pod = 1
            best_pod = alt_pod
            policy[-1] = best_pod
            POD_KV_CACHE[best_pod].update(block_keys)

    return policy

# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return prefix_aware_router