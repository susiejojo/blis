# # generate_requests_once.py
from dataclasses import dataclass
import pickle
import random
import os

random.seed(42)
from request_types import InferenceRequest

def random_words(n):
    return " ".join(random_word() for _ in range(n))

def random_word(min_len=3, max_len=3):
    import string
    letters = string.ascii_lowercase
    return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

def create_repeated_req_per_simulator(
    num_sims=4,
    num_reqs=200,
    content_len=5000,
    cache_dir="req_cache",
):
    """
    Create a request stream where:
    - Most requests are unique
    - Every (num_sims + 1)-th request is an exact duplicate of the previous one
      (used to test prefix caching + routing stickiness)

    Requests are cached to disk (pickle) and reused if available.
    """

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir,
        f"requests_sims{num_sims}_reqs{num_reqs}_len{content_len}.pkl"
    )

    # ---- Load from cache if exists ----
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("retruning from the cache")
            return pickle.load(f)

    # ---- Otherwise generate requests ----
    random.seed(42)
    shared_body = random_word() + " "
    output_word = random_word()

    requests = []
    t = 0.0
    counter = 1

    while len(requests) < num_reqs:
        # Exact duplicates every (num_sims + 1)-th request
        if counter % (num_sims + 1) == 0:
            prompt = f"{counter - 1} " + (shared_body * content_len)
        else:
            prompt = f"{counter} " + (shared_body * content_len)

        requests.append(
            InferenceRequest(
                arrival_time=t,
                input=prompt,
                output=output_word,
            )
        )

        t += 1
        counter += 1

    # ---- Save to cache ----
    with open(cache_path, "wb") as f:
        pickle.dump(requests, f)

    return requests

def create_repeated_req_per_simulator_w_suffix(
    num_sims=4,
    num_reqs=200,
    content_len=5000,
    reqs_per_sec=1.0,
    cache_dir="req_cache_suffix",
):
    """
    Create a request stream where:
    - Most requests are unique
    - Every (num_sims + 1)-th request is an exact duplicate of the previous one
      (used to test prefix caching + routing stickiness)

    Requests are cached to disk (pickle) and reused if available.
    """

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir,
        f"requests_sims{num_sims}_reqs{num_reqs}_len{content_len}_rps{reqs_per_sec}.pkl"
    )

    # ---- Load from cache if exists ----
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("retruning from the cache")
            return pickle.load(f)

    # ---- Otherwise generate requests ----
    random.seed(42)
    shared_body = random_word() + " "
    output_word = random_word()

    requests = []
    t = 0.0
    counter = 1
    delta_t = 1.0 / reqs_per_sec
    num_repeats = 40

    while len(requests) < num_reqs:
        # Exact duplicates every (num_sims + 1)-th request
        if counter % (num_sims + 1) == 0:
            back = random.randint(1, num_sims)
            prefix = " ".join([str(counter - back)] * num_repeats)
            prompt = prefix + " " +  (shared_body * content_len)
        else:
            prefix = " ".join([str(counter)] * num_repeats)
            prompt = prefix + " " + (shared_body * content_len)
            

        requests.append(
            InferenceRequest(
                arrival_time=t,
                input=prompt + random_words(random.randint(40, 100)),
                output=output_word,
            )
        )

        t += delta_t
        counter += 1

    # ---- Save to cache ----
    with open(cache_path, "wb") as f:
        pickle.dump(requests, f)

    return requests


NUM_SIMS = 2
REQUESTS = create_repeated_req_per_simulator_w_suffix(num_reqs=100, content_len=2000, num_sims=NUM_SIMS)



# def random_word(min_len=3, max_len=10):
#     import string
#     letters = string.ascii_lowercase
#     return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

# def random_words(n):
#     return " ".join(random_word() for _ in range(n))

# def generate_requests_dummy(n=100, seed=42):
#     """
#     - Prefix length: range of WORDS
#     - Prefix words: completely random
#     - High prefix reuse to show routing benefit
#     - ~10–20 req/s (bursty arrivals)
#     """
#     random.seed(seed)

#     print("creating random requests of", n)

#     num_prefixes = 20  # SMALL pool → heavy reuse
#     prefix_pool = []

#     # ---- Create reusable random prefixes ----
#     for _ in range(num_prefixes):
#         prefix_len = random.randint(50, 200)
#         prefix_pool.append(random_words(prefix_len))

#     requests = []
#     t = 0.0

#     for i in range(n):
#         # Heavy prefix reuse
#         prefix = random.choice(prefix_pool)

#         # Long, variable suffix (noise / continuation)
#         suffix_len = random.randint(10, 60)
#         suffix = random_words(suffix_len)

#         req = InferenceRequest(
#             arrival_time=t,
#             input=prefix + " " + suffix,
#             output=random_words(random.randint(2, 5))
#         )

#         requests.append(req)

#         # Inter-arrival time
#         # Mean ≈ 0.07–0.1 sec → ~10–15 req/s
#         t += random.expovariate(20)

#     return requests


# REQUESTS = generate_requests_dummy(400)

# with open("requests.pkl", "wb") as f:
#     pickle.dump(REQUESTS, f)

# print("Saved", len(REQUESTS), "requests")


# def generate_requests(n=10, start_time=0.0):
#     """
#     - Read prompts from promptsblisopenevolve.txt (separated by blank lines)
#     - Generate arrival times (~5 req/sec)
#     - Random output lengths [16, 256]
#     """

#     def read_prompts(path):
#         with open(path, "r") as f:
#             text = f.read()

#         return [
#             p.strip()
#             for p in text.split("\n\n")
#             if p.strip()
#         ]
#     print("--now here")
#     prompts = read_prompts("data/promptsblisopenevolve.txt")
#     print("--now here2")
#     assert prompts, "No prompts found!"

#     requests = []
#     current_time = start_time

#     for i in range(n):
#         # inter-arrival time (Poisson process)
#         current_time += random.expovariate(5)  # avg 5 req/sec

#         # pick a prompt (cycle if n > num_prompts)
#         prompt = prompts[i % len(prompts)]

#         output_len = random.randint(16, 64)

#         inp = prompt
#         out = "x " * output_len  # simple synthetic output

#         req = InferenceRequest(
#             arrival_time=round(current_time, 6),
#             input=inp,
#             output=out.strip(),
#         )
#         requests.append(req)

#     return requests

# def random_word(min_len=3, max_len=10):
#     import string
#     letters = string.ascii_lowercase
#     return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

# def random_words(n):
#     return " ".join(random_word() for _ in range(n))

# # def generate_requests_dummy(n=100):
# #     """
# #     - Prefix length: 23–128 WORDS
# #     - Prefix words: completely random
# #     - High prefix reuse to show routing benefit
# #     """
# #     # Create a small pool of reusable random prefixes
# #     print("creating random requests of ", n)
# #     num_prefixes = 5
# #     prefix_pool = []

# #     for _ in range(num_prefixes):
# #         prefix_len = random.randint(23, 128)
# #         prefix_pool.append(random_words(prefix_len))

# #     requests = []
# #     t = 0.0

# #     for _ in range(n):
# #         prefix = random.choice(prefix_pool)

# #         # Long, variable suffix (noise / continuation)
# #         suffix_words = random.randint(50, 200)
# #         suffix = random_words(suffix_words)

# #         req = InferenceRequest(
# #             arrival_time=t,
# #             input=prefix + " " + suffix,
# #             output=""
# #         )

# #         requests.append(req)
# #         t += random.expovariate(8)  # bursty arrivals

# #     return requests

# # def generate_requests_dummy(n=300):
#     print("creating random requests of ", n)

#     num_prefixes = 4
#     prefix_pool = [
#         random_words(random.randint(100, 200))
#         for _ in range(num_prefixes)
#     ]

#     requests = []
#     t = 0.0

#     burst_size = 25
#     for prefix in prefix_pool:
#         for _ in range(burst_size):
#             suffix = random_words(random.randint(10, 30))

#             requests.append(
#                 InferenceRequest(
#                     arrival_time=t,
#                     input=prefix + " " + suffix,
#                     output=""
#                 )
#             )

#             t += random.expovariate(20)  # tight burst

#         t += 2.0  # gap between prefix bursts

#     return requests[:n]

# def generate_requests_dummy():
#     with open("requests.pkl", "rb") as f:
#         return pickle.load(f)
    

#     ###

# def extract_prefix(request, n_words=20):
#     return " ".join(request.input.split()[:n_words])
    
# def router_rand(requests, num_sims=2):
#     """
#     Returns a list of simulator IDs, one per request.
#     """
#     policy = []
#     for req in requests:
#         # initial dumb policy (random)
#         policy.append(random.randint(0, num_sims - 1))
#     return policy

# def router_prefix(requests, num_sims=2, n_prefix_words=50):
#     policy = []
#     for req in requests:
#         prefix = extract_prefix(req, n_prefix_words)
#         h = int(hashlib.sha256(prefix.encode()).hexdigest(), 16)
#         sim_id = h % num_sims
#         policy.append(sim_id)
#     return policy

# def router_rrb(requests, num_sims=2):
#     """
#     Returns a list of simulator IDs, one per request.
#     """
#     policy = []
#     current_sim_idx = 0 # Initialize a counter for round-robin routing
#     for req in requests:
#         # The previous hash-based routing failed due to 'unhashable type: InferenceRequest'.
#         # Implementing a simple round-robin policy to ensure even load distribution across simulators.
#         # This strategy provides a basic form of load balancing and avoids the hashability issue.
#         sim_id = current_sim_idx % num_sims
#         policy.append(sim_id)
#         current_sim_idx += 1 # Move to the next simulator for the next request
#     return policy

# def router_prefix_lb(requests, num_sims=2, n_prefix_tokens=20):
#     """
#     Prefix-aware router with soft load balancing.
    
#     - Same prefix → same preferred simulator
#     - If overloaded, spill to next simulator
#     - Uses only num_sims (no simulator objects)
#     """
#     policy = []

#     # Track simulated load per simulator
#     sim_load = defaultdict(int)

#     # Cache prefix → simulator mapping for stickiness
#     prefix_to_sim = {}

#     for req in requests:
#         prefix = extract_prefix(req, n_prefix_tokens)

#         # If prefix already assigned, use it
#         if prefix in prefix_to_sim:
#             sim_id = prefix_to_sim[prefix]
#         else:
#             # Deterministic starting point from prefix hash
#             h = int(hashlib.sha256(str(prefix).encode()).hexdigest(), 16)
#             start = h % num_sims

#             # Soft load balancing: probe simulators in round-robin order
#             candidates = [(start + i) % num_sims for i in range(num_sims)]
#             sim_id = min(candidates, key=lambda i: sim_load[i])

#             prefix_to_sim[prefix] = sim_id

#         policy.append(sim_id)
#         sim_load[sim_id] += 1

#     return policy

