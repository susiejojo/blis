# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # Route based on request input for locality and deterministic assignment.
        # Assuming request.input is hashable (e.g., string, tuple, number).
        # Requests with the same input will always go to the same simulator.
        sim_id = hash(req.input) % num_sims
        policy.append(sim_id)
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
