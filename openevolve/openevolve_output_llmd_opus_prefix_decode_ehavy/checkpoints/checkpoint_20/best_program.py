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
        
        # 3. Enhanced scoring with partial matches and load awareness
        best_pods = []
        best_score = -1
        pod_scores = {}
        
        # Track request counts for load balancing
        pod_loads = [0] * num_sims
        for p in policy:
            pod_loads[p] += 1
        
        for pod_id, pod_cache in POD_KV_CACHE.items():
            # Count both prefix and total matches
            prefix_score = 0
            total_matches = 0
            
            # Prefix scoring (strict ordering)
            for key in block_keys:
                if key in pod_cache:
                    prefix_score += 1
                else:
                    break
            
            # Total matches (any position)
            for key in block_keys:
                if key in pod_cache:
                    total_matches += 1
            
            # Combined score: prioritize prefix but consider total matches
            # Add small penalty for overloaded pods
            load_penalty = 0.1 * (pod_loads[pod_id] / max(1, len(policy)))
            combined_score = prefix_score + 0.2 * (total_matches - prefix_score) - load_penalty
            
            pod_scores[pod_id] = (combined_score, prefix_score, total_matches)
            
            if combined_score > best_score:
                best_score = combined_score
                best_pods = [pod_id]
            elif abs(combined_score - best_score) < 0.01:  # Allow near-ties
                best_pods.append(pod_id)
        
        # Enhanced tie-breaking: consider multiple factors
        if len(best_pods) > 1:
            # Sort by: 1) actual prefix score, 2) load, 3) cache efficiency
            def tie_break_score(p):
                _, prefix, total = pod_scores[p]
                load = pod_loads[p]
                cache_size = len(POD_KV_CACHE[p])
                # Prefer: high prefix, low load, efficient cache usage
                return (-prefix, load, cache_size / max(1, total))
            
            best_pods.sort(key=tie_break_score)
            best_pod = best_pods[0]
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

        # 5. Update KV cache AFTER routing
        POD_KV_CACHE[best_pod].update(block_keys) # DONT CHANGE HERE!!!

    return policy

# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return prefix_aware_router