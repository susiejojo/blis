# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # New policy: Deterministic routing based on request content hash.
        # This assumes that requests can be meaningfully distinguished by their string representation,
        # and that consistent routing for similar requests can improve average reward.
        # The `abs()` ensures the hash modulo result is non-negative, yielding a valid simulator ID.
        routing_key = hash(str(req))
        sim_id = abs(routing_key) % num_sims
        policy.append(sim_id)
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
