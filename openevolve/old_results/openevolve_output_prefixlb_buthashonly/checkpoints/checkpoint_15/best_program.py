# EVOLVE-BLOCK-START
import random

def router(requests, num_sims=2):
    """
    Returns a list of simulator IDs, one per request.
    """
    n_requests = len(requests)
    if n_requests == 0:
        return []

    policy = [0] * n_requests 
    
    # Stores preferred simulator ID for each request index
    # and also a list of indices for each preferred simulator
    preferred_assignments_per_request_idx = [0] * n_requests
    preferred_groups_by_sim = [[] for _ in range(num_sims)]

    for i, req in enumerate(requests):
        # Calculate preferred simulator based on hash for locality
        preferred_sim_id = hash(req.input) % num_sims
        preferred_assignments_per_request_idx[i] = preferred_sim_id
        preferred_groups_by_sim[preferred_sim_id].append(i)

    # Calculate desired average load per simulator (ceiling for fair distribution)
    base_load_per_sim = n_requests // num_sims
    
    # To manage load distribution, we'll keep track of actual assignments made
    actual_sim_loads = [0] * num_sims
    
    # Store indices of requests that need to be reassigned because their
    # preferred simulator became "overloaded" early in the assignment process.
    reassign_candidates = []

    # First pass: Assign requests greedily to their preferred simulator,
    # up to a certain threshold (max_allowed_for_locality).
    # Requests beyond this threshold become candidates for rebalancing.
    for sim_id in range(num_sims):
        group = preferred_groups_by_sim[sim_id]
        
        # Sort requests within each group by their original index for deterministic behavior
        group.sort() 
        
        # Calculate the maximum number of requests this simulator can take from its preferred group
        # This distributes the remainder (n_requests % num_sims) evenly among the first few simulators
        max_allowed_for_locality = base_load_per_sim + (1 if sim_id < (n_requests % num_sims) else 0)

        # Assign requests to their preferred sim if capacity allows
        for i, req_idx in enumerate(group):
            if actual_sim_loads[sim_id] < max_allowed_for_locality: 
                policy[req_idx] = sim_id
                actual_sim_loads[sim_id] += 1
            else: # These requests exceed the locality limit for this sim, mark for rebalancing
                reassign_candidates.append(req_idx)

    # Step 3: Reassign candidates to the least loaded simulators.
    # Shuffle the reassign candidates to break any potential patterns from original ordering
    random.shuffle(reassign_candidates) 
    
    for req_idx in reassign_candidates:
        # Find the simulator with the minimum current load
        least_loaded_sim = actual_sim_loads.index(min(actual_sim_loads))
        policy[req_idx] = least_loaded_sim
        actual_sim_loads[least_loaded_sim] += 1
            
    return policy
# EVOLVE-BLOCK-END

def run_search():
    # required entry point for evaluator
    return router
