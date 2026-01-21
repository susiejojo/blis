# EVOLVE-BLOCK-START
def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    # Initialize persistent state for round-robin.
    # This state will persist across calls to the 'router' function instance.
    if not hasattr(router, 'next_sim_idx'):
        router.next_sim_idx = 0
    
    policy = []
    for req in requests:
        # Assign current request to the next simulator in a round-robin fashion.
        sim_id = router.next_sim_idx
        policy.append(sim_id)
        
        # Update the next_sim_idx for the subsequent request.
        # This handles cases where num_sims might change between calls.
        router.next_sim_idx = (router.next_sim_idx + 1) % num_sims
        
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
