# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    current_sim_idx = 0 # Initialize a counter for round-robin routing
    for req in requests:
        # The previous hash-based routing failed due to 'unhashable type: InferenceRequest'.
        # Implementing a simple round-robin policy to ensure even load distribution across simulators.
        # This strategy provides a basic form of load balancing and avoids the hashability issue.
        sim_id = current_sim_idx % num_sims
        policy.append(sim_id)
        current_sim_idx += 1 # Move to the next simulator for the next request
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
