"""
- Computes the Greedy, RIS and RIS-Exact solutions to the IM problem.
- Designed to run on AWS GPU instances.
- Expects arguments via terminal entry as descrbed in the CommanLine() class below:
- Output is stored in a csv results file detailing the model run parameters and the resulting seed sets and spreads.
"""

import subprocess, argparse, time, os, re, csv
import pandas as pd
from function_file import *

# ---------------
# Define command line argument class
# ---------------

class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description = "The keyword arguments are:")
        parser.add_argument("-t", help = "Type of network to run - (ER, WS, SF)", required = True)
        parser.add_argument("-p", help = "Real number on (0,1) - propagation probability", required = True)
        parser.add_argument("-k", help = "Integer - number of seed nodes", required = True)
        parser.add_argument("-th", help = "Integer - number of RRR sets to produce (theta)", required = True)
        parser.add_argument("-mcg", help = "Integer - number of MC iterations for greedy procedure", required = True)
        parser.add_argument("-n", help = "Integer - limit number of files to run algorithm on", required = False, default = None)

        arg = parser.parse_args()

        print("Algorithm running on {0} networks".format(arg.t))
        print("Propagation probability set to {0}".format(arg.p))
        print("Seed set size set to {0}".format(arg.k))
        print("{0} RRR sets will be created".format(arg.th))
        print("Greedy algorithm running with {0} iterations".format(arg.mcg))
        if arg.n:
            print("Algorithm limited to run on {0} different graphs".format(arg.n))
        print("")
        
        self.t, self.n, self.p = arg.t, arg.n, arg.p
        self.th, self.k = arg.th, arg.k
        self.mcg = arg.mcg

# ---------------
# Run algorithms
# ---------------

def main():

    # Read in keyword arguments 
    output_path = "./output/results.csv"
    app = CommandLine()
    t, n, p = app.t, app.n, float(app.p)
    theta, mc_greed, k = int(app.th), int(app.mcg), int(app.k)
    
    # Create output folders and files if necessary
    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists(output_path):
        pd.DataFrame(columns = ['file','network_type','k','p','seed_ris','seed_greedy',
                                'seed_exact','spread_ris','spread_greedy','spread_exact',
                                'spread_ris_ic','spread_greedy_ic','spread_exact_ic']).to_csv(output_path, index=False)

    # Get file lists from input folder and output file
    input_files  = os.listdir("./network_data/" + t) 
    d = pd.read_csv(output_path)
    d = d.loc[(d['k'] == k) & (d['p'] == p)]
    output_files = d['file'].tolist()

    # Get files that haven't yet been done
    files = list(set(input_files) - set(output_files))
    if n:
        files = files[:int(n)]
    
    # Run algorithms on each file
    for f in files:
        print("")
        print("----------------")
        print(f)
        
        # Read Graph
        fname = "./network_data/" + t + "/" + f
        G = pd.read_csv(fname).values
        
        # Extract parameters
        fileparams = re.split('(\d+)',f)
        nodes = int(fileparams[1])

        # Construct R, the set of RRR sets
        print("Constructing collection of RRR sets (R) ...")
        R = get_rrs_gpu(G, nodes, p, theta)
        
        # Run RIS
        print("Running RIS ...")
        seeds_ris, sp_ris = ris(R,k)
        print(seeds_ris)
        print(sp_ris)
        print("")
        
        # Run RIS-Exact
        print("Running RIS-Exact...")
        start = time.time()
        seeds_exact, sp_exact = ris_complete_gpu(R, k)
        print("Runtime: {0} seconds".format(time.time() - start))
        print(seeds_exact)
        print(sp_exact)
        print("")

        # Get Greedy
        print("Running Greedy...")
        start = time.time()
        graph = load_graph(fname, nodes)
        seeds_greed = greedy_gpu(graph, k, p, mc_greed)
        sp_greed = get_spread(R,seeds_greed)
        print("Runtime: {0} seconds".format(time.time() - start))
        print(seeds_greed)
        print(sp_greed)
        print("")

        # Get the spreads of each algorithm via the IC function
        print("Running IC spread evaluations...")
        S_ris = np.full(nodes, False, dtype = bool)
        S_ris[seeds_ris] = True
        S_exact = np.full(nodes, False, dtype = bool)
        S_exact[seeds_exact] = True
        S_greed = np.full(nodes, False, dtype = bool)
        S_greed[seeds_greed] = True

        sp_exact_ic = IC_gpu(graph, S_exact, p, mc_greed) / float(nodes)
        sp_ris_ic   = IC_gpu(graph, S_ris, p, mc_greed) / float(nodes)
        sp_greed_ic = IC_gpu(graph, S_greed, p, mc_greed) / float(nodes)
     
        # Gather into dataframe
        results = [f,t,k,p,seeds_ris,seeds_greed,seeds_exact,sp_ris,sp_greed,sp_exact,sp_ris_ic,sp_greed_ic,sp_exact_ic]
        with open(output_path, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(results)
        
if __name__ == '__main__':
    main()
