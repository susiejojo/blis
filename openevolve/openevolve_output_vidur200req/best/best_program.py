# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # initial dumb policy (random)
        policy.append(random.randint(0, num_sims - 1))
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
