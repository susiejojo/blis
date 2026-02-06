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

        # 3. Longest-prefix scoring (inline) with threshold and tie-break on pod_id parity
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
            # Apply a threshold of 2 to encourage spillover when prefix matches are weak
            if score >= 2:
                if score > best_score:
                    best_score = score
                    tie_candidates = [pod_id]
                elif score == best_score:
                    tie_candidates.append(pod_id)

        if best_score < 2:
            # For low prefix matches, choose pod with smallest cache size to balance load
            sizes = {pid: len(cache) for pid, cache in POD_KV_CACHE.items()}
            min_size = min(sizes.values())
            candidates = [pid for pid, size in sizes.items() if size == min_size]
            best_pod = candidates[len(policy) % len(candidates)]
        else:
            # Tie-break by preferring even pod_ids first, then odd, cycling through candidates
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

    # Additional: enforce prefix dominance for pod 0 only if prefix matches >= 4 and best_pod != 0 with threshold jitter and flip
    prefix_length_threshold = 4
    for idx, req in enumerate(requests):
        current_pod = policy[idx]
        tokens = tokenize_prompt(req.input, tokenizer)
        block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)
        pod0_cache = POD_KV_CACHE[0]
        common_prefix = 0
        for key in block_keys:
            if key in pod0_cache:
                common_prefix += 1
            else:
                break
        if current_pod != 0:
            adjusted_threshold = prefix_length_threshold + (idx % 3 == 0)
            # Flip dominance occasionally to diversify routing behavior
            dominance_flip = (idx % 5 == 0)
            if (common_prefix >= adjusted_threshold) != dominance_flip:
                # Override routing to pod 0 for prefix dominance
                policy[idx] = 0
                POD_KV_CACHE[0].update(block_keys)
        else:
            # Occasionally force pod 0 to relinquish dominance even if prefix is high, to force spillover
            spill_condition = (common_prefix >= prefix_length_threshold) and (idx % 7 == 0)
            if spill_condition and num_sims > 1:
                alt_pod = 1 + (idx % (num_sims - 1))
                policy[idx] = alt_pod
                POD_KV_CACHE[alt_pod].update(block_keys)

    # Additional: forcibly spillover traffic from pod 0 to alternating pods every 3rd request to prevent overload
    if num_sims > 1:
        for i in range(len(policy)):
            if policy[i] == 0 and (i + 1) % 3 == 0:
                tokens = tokenize_prompt(requests[i].input, tokenizer)
                block_keys = tokens_to_kv_block_keys(tokens, kv_block_size)
                # Alternate spillover pod between pod 1 and pod 2 (if present) to balance load better
                alt_pod = 1 + ((i // 3) % (num_sims - 1))
                policy[i] = alt_pod
                POD_KV_CACHE[alt_pod].update(block_keys)

    return policy

# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return prefix_aware_router