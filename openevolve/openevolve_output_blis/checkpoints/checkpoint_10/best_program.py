# EVOLVE-BLOCK-START
def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    This improved router balances load while preserving request locality based on input prefixes.
    """
    policy = []
    
    # Store mappings from request prefixes to simulator IDs to preserve locality.
    # This allows requests with shared prefixes to benefit from being routed to the same simulator.
    prefix_to_sim_map = {}
    
    # Track the current number of requests assigned to each simulator.
    # This helps with load balancing when assigning new prefixes.
    sim_loads = [0] * num_sims 

    for req in requests:
        # Extract the prefix for locality. The problem hints at "first few blocks"
        # and "a block is often 16 tokens". We'll use the first 16 elements/characters.
        prefix_length = 16
        # Slice the input to get the prefix segment.
        prefix_segment = req.input[:min(len(req.input), prefix_length)]
        # Convert the prefix segment to a hashable type (e.g., tuple) to use as a dictionary key.
        # This handles both string inputs (tuple of chars) and list of token inputs (tuple of tokens).
        prefix = tuple(prefix_segment) 

        sim_id = -1
        if prefix in prefix_to_sim_map:
            # If this prefix has been seen before, route to the same simulator
            # to preserve locality and leverage potential simulator state/cache.
            sim_id = prefix_to_sim_map[prefix]
        else:
            # If it's a new prefix, assign it to the simulator with the minimum current load.
            # This helps balance the load across simulators for unseen prefixes,
            # ensuring no single simulator becomes a bottleneck prematurely.
            min_load = float('inf')
            min_load_sim_id = -1
            for i in range(num_sims):
                if sim_loads[i] < min_load:
                    min_load = sim_loads[i]
                    min_load_sim_id = i
            
            sim_id = min_load_sim_id
            prefix_to_sim_map[prefix] = sim_id # Store the mapping for future requests with this prefix
        
        policy.append(sim_id)
        sim_loads[sim_id] += 1 # Increment the load for the assigned simulator
        
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
