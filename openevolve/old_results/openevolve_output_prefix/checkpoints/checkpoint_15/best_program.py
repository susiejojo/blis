# EVOLVE-BLOCK-START
import random # Not used in core routing logic here, but kept for consistency or potential future variations.

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    This version aims to maximize locality by routing requests with shared prefixes
    to the same simulator, while also balancing the load across simulators
    by assigning new prefixes to the least loaded simulator at that point in the batch.

    The prefix (routing_key) is extracted from request.input:
    - The request.input string up to the last '/' character, if present.
    - The entire request.input string, if no '/' is found.
    """
    policy = []
    
    # Maps extracted routing keys (prefixes) to simulator IDs.
    # Ensures all requests with the same routing_key go to the same simulator.
    prefix_to_sim_map = {} 

    # Tracks the number of requests assigned to each simulator in the current batch.
    # Used to make load-aware decisions for new prefixes.
    sim_loads = [0] * num_sims

    for req in requests:
        input_data = str(req.input)

        # Determine the routing key (prefix)
        # This approach seeks to group requests that share a 'parent' identifier
        # (e.g., 'user/123/profile' and 'user/123/orders' would both use 'user/123' as the key).
        last_slash_idx = input_data.rfind('/')
        if last_slash_idx != -1:
            routing_key = input_data[:last_slash_idx]
        else:
            # If no slash, the entire input string is considered the key.
            routing_key = input_data

        assigned_sim_id = None
        if routing_key in prefix_to_sim_map:
            # If prefix seen before, use its assigned simulator to preserve locality.
            assigned_sim_id = prefix_to_sim_map[routing_key]
        else:
            # If new prefix, assign it to the currently least loaded simulator.
            # This helps balance load as new prefixes are introduced.
            least_loaded_sim_id = 0
            min_load = sim_loads[0]
            for i in range(1, num_sims):
                if sim_loads[i] < min_load:
                    min_load = sim_loads[i]
                    least_loaded_sim_id = i
            
            assigned_sim_id = least_loaded_sim_id
            prefix_to_sim_map[routing_key] = assigned_sim_id
        
        # Update load for the assigned simulator
        sim_loads[assigned_sim_id] += 1
        policy.append(assigned_sim_id)
        
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
