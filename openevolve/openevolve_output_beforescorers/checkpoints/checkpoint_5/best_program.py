# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # Route requests based on input prefix for locality.
        # A block is often 16 tokens.
        prefix_length = 16
        prefix_data = None

        # Determine the hashable prefix from req.input
        if isinstance(req.input, (list, tuple)):
            # If req.input is a sequence of tokens (list or tuple),
            # slice it and convert to a tuple for hashing.
            prefix_data = tuple(req.input[:min(prefix_length, len(req.input))])
        elif isinstance(req.input, str):
            # If req.input is a string, slice it directly.
            prefix_data = req.input[:min(prefix_length, len(req.input))]
        else:
            # Fallback: for other types, use its string representation for hashing.
            # This ensures hashability for unexpected types.
            prefix_data = str(req.input)
            
        # Assign to a simulator based on the hash of the prefix
        sim_id = hash(prefix_data) % num_sims
        policy.append(sim_id)
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
