# IM-Evaluation

Repository contains Python code used to generate the results in the paper entitled [*A Numerical Evaluation of the Accuracy of Influence Maximization Algorithms*](https://hautahi.com/static/docs/IM_Accuracy_Paper.pdf) by Kingi, Wang, Shafer et al., which has the following abstract:

*We develop an algorithm to compute exact solutions to the influence maximization problem using concepts from reverse influence sampling (RIS). We implement the algorithm using GPU resources to evaluate the empirical accuracy of theoretically-guaranteed greedy and RIS approximate solutions. We find that the approximation algorithms yield solutions that are remarkably close to optimal - usually achieving greater than 99% of the optimal influence spread. These results are consistent across a wide range of network structures.*

## Python File Descriptions:

- `function_file.py` defines the various functions required to implement the IM algorithms described in the paper on a GPU architecture.

- `1.create_networks.py` is self-contained, and generates the network csv files used in the paper, which are stored in the `network_data` folder. The optional keyword flags are `-n`, which is the number of nodes in the generated networks, and `-vers`, which is the number of versions of each graph to create. Run this first, in the usual fashion:

    `python3 1.create_networks.py -n 100 -vers 10`

    where the values given above are the default values used in the paper.

- `2.run_simulations.py` conducts the analyses. It calls the functions defined in `function_file.py`. Run it from the command line as follows:

    `python3 2.run_simulations.py -t SF -p 0.01 -k 4 -th 100000 -mcg 100000 -n 100`

    where `-t` is either 'SF' 'ER' or 'WS' depending on which network type is desired,`-p` specifies the propagation probability, `-k` the seet set size, `-th` the number of RRR sets to generate for the RIS procedures, `-mcg` the number of MC iterations to perform to compute the spread of the IC function, and `-n` an optional parameter that allows the user to specify a maximum number of graphs to simulate in one run. This file needs to be run several times (one for each network type and propagation probability combination) to generate the results in the paper. This takes approximately 4 days. The results from these runs are stored in the `./output/results.csv` file.

- `3.make_graphs.ipynb` is a Jupyter notebook that produces the two graphs used in the paper as well as a number of exploratory analyses. The graphs are saved in the `output` folder.

## AWS Instructions

The code was run on an AWS instance. The below notes were useful reminders to myself.

1. Launch instance on the AWS website: Deep Learning Base AMI (Amazon Linux) Version 19.1 (ami-00a1164673faf2ac3), p2.xlarge
2. Login to AWS instance via: `ssh -i path/to/amazonkey.pem ec2-user@instance-address.amazonaws.com`
3. Setup AWS instance with: `sudo pip3 install numba`
4. Transfer file to instance: `scp -i amazonkey.pem file_name ec2-user@instance-address.amazonaws.com:`
5. Transfer folder to instance: `scp -i amazonkey.pem -r folder_name ec2-user@instance-address.amazonaws.com:`
6. Transfer files back to local machine: `scp -i amazonkey.pem -r ec2-user@instance-address.amazonaws.com: .`
7. Tip: Use `tmux` command before running a script to open a new screen. Transition back to main screen with `ctrl+b,d` and then back again using `tmux attach -d`. This allows you to log out of AWS while keeping a script running.
