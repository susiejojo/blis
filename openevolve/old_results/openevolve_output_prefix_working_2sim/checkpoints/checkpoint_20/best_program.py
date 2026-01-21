# EVOLVE-BLOCK-START
def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    policy = []
    for req in requests:
        # Use hashing of request.input for locality.
        # This routes requests with identical inputs to the same simulator,
        # leveraging potential benefits from state reuse or caching,
        # as suggested by the problem statement for maximizing average reward.
        sim_id = hash(req.input) % num_sims
        policy.append(sim_id)
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
