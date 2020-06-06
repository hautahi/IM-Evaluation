"""
- Defines all functions needed to compute the Greedy, RIS and RIS-Exact solutions to the IM problem.
- Functions called from the 2.run_simulations.py file.
"""

from numba import cuda, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import pandas as pd
import math, time, itertools, sys
from numpy import float32
from os import listdir
from collections import Counter

# ---------------
# Define Algorithms to construct R - the collection of RRR sets
# ---------------

# Function creates a large number of RRR sets (to compile the "R" set)
def get_rrs_gpu(graph, node_num, p, theta):  # Do we need node_num?
    
    # Randomly choose theta number of nodes with replacement from network (theta is number of RRS sets as in paper)
    sources = np.random.choice(node_num, size=theta)

    # Initiate a mc x n matrix to represent R
    rrs = np.full((theta, node_num), False, dtype=bool)

    # Creates mc random number generator states
    rng_states = create_xoroshiro128p_states(theta,seed = np.random.randint(theta))
        
    # Number of blocks (each with 128 threads) needed to perform mc operations
    threads_per_block = 128
    blocks = math.ceil(theta / threads_per_block)

    # Update the mc rows of the rrs array using GPU
    get_node_flow_gpu[blocks, threads_per_block](graph, sources, p, rrs, theta, rng_states)

    return(rrs)

# Jit function just modifies the objects
"""
Things to know:
    - the below function is a "kernel"
    - kernels cannot explicitly return values. all result data must be written to an array passed to the function
    - kernels explicitly declare their thread hierarchy when called (see the square brackets above)
    - The product of the two numbers in the square brackets gives the total number of threads launched (mc)
    - block size is crucial:
        - On the software side, it determines how many threads share a given area of shared memory
        - On the hardware side, it must be large enough for full occupation of execution units

    - the @cuda.jit thing is known as a "decorator"
    - it marks a function for optimization by numba's JIT compiler
    - the code is executed by every thread once, so it has to know which thread it's in to know which array element(s) of rrs it is responsible for
    - this can be done manually, but easier to do automatically using cuda.grid(1), which returns the absolute position of the current thread in the entire grid of blocks
"""

@cuda.jit
def get_node_flow_gpu(graph, sources, p, rrs, theta, rng_states):
    
    # Get abosolute position of current thread
    thread_id = cuda.grid(1)
    
    # Because of block sizes, some of the threads will be greater than theta and not needed
    if thread_id >= theta:
        return
    
    # For the row in the matrix corresponding to this thread, mark the source node for the rrs set
    rrs[thread_id][sources[thread_id]] = True

    # Allocate an array for thread to use temporarily. The size of the array needs to be hard-coded!
    # We'll use this to keep track of the nodes that were newly found in each iteration/traverse of the rrs construction
    new_nodes = cuda.local.array(100, boolean)
    new_nodes[sources[thread_id]] = True
    # Construct the RRS set by traversing to find new nodes. Stop when no new nodes have been found.
    done = False
    while not done:
        done = True
        # Iterate through the new_nodes, looking for nodes that flow into them
        for source in range(new_nodes.shape[0]):
            # Only evaluate a node if it is new
            if new_nodes[source]:
                # Iterate through every edge in the graph to find the nodes that can influence the current node
                for edge in range(graph.shape[0]):
                    if graph[edge][1] == source:
                        neighbor = graph[edge][0]
                        # If a neighbor node not already in the RRS and if the edge still exists, then there is a path from neighbor to source and will be added
                        # to both the new_nodes and rrs arrays. If at least one new node is found, the while loop continues
                        if (not rrs[thread_id][neighbor]) & (p > xoroshiro128p_uniform_float32(rng_states, thread_id)):
                            new_nodes[neighbor], rrs[thread_id][neighbor], done = True, True, False
                
                # Once all the neighbors of a node have been discovered, it will no longer be evaluated
                new_nodes[source] = False

# ---------------
# Define RIS algorithm to get solution with greedy max coverage algorithm
# ---------------

def ris(R, k):
        
    # Extract indices of nodes in each rrs
    R_copy = [np.where(r)[0] for r in R]
    
    # Choose nodes that appear most often in R (maximum coverage greedy algorithm)
    seeds, count = [], 0
    for _ in range(k):

        # Create large list of all occurences of nodes in R
        flat_list = [item for sublist in R_copy for item in sublist]

        if flat_list:
            
            # Find node that occurs most often
            seed = Counter(flat_list).most_common()[0][0]

            # Find number of RRR sets covered
            count += Counter(flat_list).most_common()[0][1]

            # Remove RRSs containing last chosen seed 
            R_copy = [item for item in R_copy if seed not in item]

            # Add to outputs
            seeds.append(seed)

    return(seeds, float(count) / R.shape[0])

# ---------------
# Define algorithms to get RIS-Exact solution
# ---------------

def get_spread(R,seeds):

    # Extract indices of nodes in each rrs
    R_copy = [np.where(r)[0] for r in R]

    # Create list of RRSs covered by opt_solution
    coverage = [item for item in R_copy if set(item) & set(seeds)]

    return(len(coverage)/R.shape[0])

def ris_complete_gpu(R, k):

    # Construct all possible seed sets
    seed_sets = np.array(list(itertools.combinations(range(R.shape[1]),k)))
    
    # Initiate array indexed by each possible seed set
    max_val = np.zeros(seed_sets.shape[0])

    # Number of blocks (each with 128 threads) needed to perform operations on all candidate seed sets
    threads_per_block = 128
    blocks = math.ceil(seed_sets.shape[0] / threads_per_block)

    # Update each cell of max_val array using GPU
    get_max_val_gpu[blocks, threads_per_block](R, seed_sets, max_val)
    
    # Get optimal seed set
    opt_solution = seed_sets[np.argmax(max_val)].tolist()
    
    # Get coverage
    spread = get_spread(R,opt_solution)

    return(opt_solution, spread)

@cuda.jit
def get_max_val_gpu(R, seed_sets, max_val):
    
    # Get absolute position of current thread
    thread_id = cuda.grid(1)

    # Because of fixed block sizes, some of the threads won't be needed
    if thread_id >= seed_sets.shape[0]:
        return

    # For each RRR set in R, check if it contains a node from the candidate set
    for i in range(R.shape[0]):
        increment = False
        # Loop over each of the k nodes in the relevant candidate set and check if its in the RRR set 
        for j in range(seed_sets.shape[1]):
            seed = seed_sets[thread_id][j]    # Can I pull this out up the top?
            if R[i][seed]:                  # Use a set approach here?
                increment = True
                break
        if increment:                           # How about removing this part?
            max_val[thread_id] += 1

# ---------------
# Define algorithms to get Greedy solution
# ---------------

# Function to create adjacency matrix
def load_graph(file_name, size):

    # Load graph files
    df = pd.read_csv(file_name)

    # Initiate adjacency matrix 
    graph = np.full((size, size), False, dtype=bool)

    # Fill adjacency matrix based on csv file
    for _, row in df.iterrows():
        graph[row['source']][row['target']] = True

    return(graph)

def greedy_gpu(graph, k, p, mc):

    # Initiate array with entry for each node
    S = np.full(graph.shape[0], False, dtype = bool)
    
    S_nodes = []
    for _ in range(k):
        node, count = 0, 0          # changed this from node = count = 0
        for j in range(graph.shape[0]):

            # If it's already in the seed set don't worry about it
            if S[j]:                # Is this faster than an if-not command?
                continue
            
            # Throw this node in the seed set and compute the spread
            S_copy = S.copy()
            S_copy[j] = True
            s = IC_gpu(graph, S_copy, p, mc)
            
            # If the spread is greater than any previous one, then keep this node
            if s > count:
                count, node = s, j
        
        # Add to seed set and record the spread
        S[node] = True
        S_nodes.append(node)

    return(S_nodes)

def IC_gpu(graph, S, p, mc):

    # Number of blocks (each with 128 threads) needed to perform all mc iterations
    threads_per_block = 128
    blocks = math.ceil(mc / threads_per_block)

    # Creates mc random number generator states
    rng_states = create_xoroshiro128p_states(mc, seed=np.random.randint(mc))

    # Create an mc x n array of False's
    active = np.tile(S, (mc, 1))
    new_active = active.copy()

    # Another mc x n array of False's
    new_ones = np.full((mc, graph.shape[0]), False, dtype=bool)
    
    # Update active array using GPU
    find_spread_gpu[blocks, threads_per_block](graph, active, new_active, new_ones, mc, p, rng_states)
    
    return(np.count_nonzero(active) / mc)

@cuda.jit
def find_spread_gpu(graph, active, new_active, new_ones, mc, p, rng_states):
    
    # Get abosolute position of current thread
    thread_id = cuda.grid(1)

    # Because of fixed block sizes, some of the threads won't be needed
    if thread_id >= mc:
        return

    done = False
    while not done:
        done = True
        for j in range(new_active[thread_id].shape[0]):
            if new_active[thread_id][j]:
                for k in range(graph.shape[0]):
                    if graph[j][k] and p > xoroshiro128p_uniform_float32(rng_states, thread_id):
                        new_ones[thread_id][k] = True

        for j in range(new_active[thread_id].shape[0]):
            if new_ones[thread_id][j] and (not active[thread_id][j]):
                active[thread_id][j] = True
                new_active[thread_id][j] = True
                done = False
            else:
                new_active[thread_id][j] = False
            new_ones[thread_id][j] = False

