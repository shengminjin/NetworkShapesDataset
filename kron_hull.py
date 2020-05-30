'''
Reference implementation of Kronecker Hull.
Author: Shengmin Jin
For more details, refer to the paper:
Representing Networks with 3D Shapes
Shengmin Jin and Reza Zafarani
IEEE International Conference on Data Mining (ICDM), 2018

Versions: 1.0    The Original Version
          1.1    Code Optimization
          1.2    Bug fix for python subprocess hang
          1.3    Bug fix for wrong output
'''

from sys import argv
import random, os, shutil, time
import uuid
import math
import subprocess
import networkx as nx
import re
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib
import argparse

matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matlab.engine
from tqdm import tqdm
from joblib import Parallel, delayed




# create unique directory name
def name_uuid():
    name = str(uuid.uuid4())
    return name


# create subprocess for kronfit
def kronfit(kronfit_job):
    input_file_path = kronfit_job[0]
    output_file_path = kronfit_job[1]
    if not os.path.exists(output_file_path):
        cmd = 'kronfit', '-i:' + input_file_path, '-n0:2', '-gi:100', '-o:' + output_file_path
        subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()


# sample a subgraph
def random_node_sampling(directory, p, i):
    random.shuffle(nodes)
    size = int(len(nodes) * float(p) / 100)
    sample = nodes[:size]
    sub_g = G.subgraph(sample)
    sub_g = sub_g.to_directed()  # Forgot to add
    nx.write_edgelist(sub_g, directory + str(p) + '/'
                      + str(i) + '.edgelist', delimiter='\t', data=False)


def random_edge_sampling(directory, p, i):
    sub_g = G.copy()
    edges = [e for e in sub_g.edges]
    random.shuffle(edges)
    size = int(len(edges) * float(100 - p) / 100)
    sub_g.remove_edges_from(edges[:size])
    sub_g = sub_g.to_directed()  # Forgot to add
    nx.write_edgelist(sub_g, directory + str(p) + '/'
                      + str(i) + '.edgelist', delimiter='\t', data=False)


def random_walk_with_restart_sampling(directory, p, i, restart_prob=0.15, jump_iteration=10, seed=None):
    # set random seed
    random.seed(seed)

    # sample size round down to interger
    sample_size = int(len(nodes) * float(p) / 100)

    # set starting node
    startnode = random.choice(nodes)
    currentnode = startnode

    # used for jump when no new node visited in certain iteration
    restart_iteration = 0
    last_number_of_nodes = 0

    # result node set and total iteration
    nodelist = set()
    total_iteration = 0
    while len(nodelist) < sample_size:
        # add current node
        total_iteration += 1
        nodelist.add(currentnode)
        # restart with certain prob
        x = random.random()
        if x < restart_prob:
            currentnode = startnode
        else:
            # move a step forward
            nextnode = random.choice(list(G[currentnode]))
            currentnode = nextnode
        # find a new startnode if number of nodes in sample does not grow
        if restart_iteration < jump_iteration:
            restart_iteration += 1
        else:
            if last_number_of_nodes == len(nodelist):
                startnode = random.choice(nodes)
                currentnode = startnode
            restart_iteration = 0
            last_number_of_nodes = len(nodelist)
    sub_g = G.subgraph(nodelist).to_directed()
    nx.write_edgelist(sub_g, directory + str(p) + '/'
                      + str(i) + '.edgelist', delimiter='\t', data=False)


def create_kronecker_hull(directory, points, display_name):
    eng = matlab.engine.start_matlab()
    eng.get_convex_hull(matlab.double(points), directory, display_name)

def create_cuboid(directory, points, display_name):
    eng = matlab.engine.start_matlab()
    eng.get_cuboid(matlab.double(points), directory, display_name)

def create_sphere(directory, points, display_name):
    eng = matlab.engine.start_matlab()
    eng.get_sphere(matlab.double(points), directory, display_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generating a 3D Network Shapes with a Kronecker Hull')

    parser.add_argument('-n', required=False,
                        default='as20graph',
                        help='Network name')

    parser.add_argument('-f', required=False,
                        default='as20graph.txt',
                        help='Input edge list file name.')

    parser.add_argument('-s', required=False,
                        default=20,
                        help='Sampling proportion step (percentage)')

    parser.add_argument('-t', required=False,
                        default=5,
                        help='Number of samples for each sampling proporation"')

    parser.add_argument('-id', required=False,
                        default='as20graph',
                        help='Unique ID"')

    parser.add_argument('-sampling', required=False,
                        default='re',
                        help='Sampling Methods')

    parser.add_argument('-z', '--zip', required=False,
                        help='Copy and Zip certain files to a new directory for downloading',
                        action='store_true')

    #    network_name = myargs['-n']
    #    edgelist = myargs['-f']
    #    step = int(myargs['-s'])
    #    nos = int(myargs['-t'])

    args = parser.parse_args()
    uid = args.id
    network_name = 'downloads' + '/' + uid
    # network_name = 'downloads' + '/' + name_uuid()

    # print("Command: {}".format(args.command))


    edgelist = 'uploads' + '/' + args.f
    step = int(args.s)
    nos = int(args.t)
    sampling_method = args.sampling


    print("Network name: {}".format(network_name))
    print("Input edge file: {}".format(args.f))
    print("Sampling proportion step: {}".format(args.s))
    print("Number of samples for each sampling proportion: {}\n\n".format(args.t))

    directory = network_name + '/'
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    G = nx.read_edgelist(edgelist, delimiter='\t')
    nodes = list(G.nodes())

    print(edgelist)

    nx.write_edgelist(G.to_directed(), directory + '100.edgelist'
                      , delimiter='\t', data=False)

    kronfit_jobs = []

    # kronfit the original graph
    full_input_file = directory + '100.edgelist'
    full_output_file = directory + 'output.dat'
    kronfit_jobs.append((full_input_file, full_output_file))

    output = open(directory + 'kron_points.txt', 'w')
    output.write('a,b,d,sampling_proportion\n')

    # get sample graphs and Kronecker points of the samples
    for p in range(step, 100, step):
        print('Sampling ' + str(p) + '% subgraphs')
        processes = []
        os.mkdir(directory + str(p) + '/')
        for i in range(0, nos):
            # random_node_sampling(directory, p, i)
            # random_edge_sampling(directory, p, i)
            if sampling_method == 're':
                random_edge_sampling(directory, p, i)
            elif sampling_method == 'rn':
                random_node_sampling(directory, p, i)
            elif sampling_method == 'rw':
                random_walk_with_restart_sampling(directory, p, i)
            input_file = directory + str(p) + '/' + str(i) + '.edgelist'
            output_file = directory + str(p) + '/' + str(i) + '_output.dat'
            kronfit_jobs.append((input_file, output_file))

    print("Running Kronfit for each graph")

    Parallel(n_jobs=int((len(os.sched_getaffinity(0)) / 2)))(
        delayed(kronfit)(kronfit_job)
        for kronfit_job in tqdm(kronfit_jobs))

    print("Kronfit Finished")

    kronecker_points = []
    for p in range(step, 100, step):
        # extract points from the output file of kronfit
        for i in range(0, nos):
            output_file = directory + str(p) + '/' + str(i) + '_output.dat'
            with open(output_file, 'r') as myfile:
                s = myfile.read()
                ret = re.findall(r'\[([^]]*)\]', s)
                split = ret[0].split(',')
                a = split[0]
                b = split[1].split(';')[0].strip()
                d = split[2].strip()
                output.write(str(a) + ',' + str(b) + ',' + str(d) + ',' + str(p) + '\n')
                kronecker_points.append([float(a), float(b), float(d)])

    # write all the Kronecker points to a file
    with open(full_output_file, 'r') as myfile:
        s = myfile.read()
        ret = re.findall(r'\[([^]]*)\]', s)
        split = ret[0].split(',')
        a = split[0]
        b = split[1].split(';')[0].strip()
        d = split[2].strip()
        output.write(str(a) + ',' + str(b) + ',' + str(d) + ',' + str(100) + '\n')
        kronecker_points.append([float(a), float(b), float(d)])

    output.close()

    create_kronecker_hull(directory, kronecker_points, args.n)
    create_cuboid(directory, kronecker_points, args.n)
    create_sphere(directory, kronecker_points, args.n)

    if args.zip:
        # makes new directory network_shape and copies them to it
        os.mkdir(directory + '/' + 'network_shape' + '/')
        shutil.copy2(directory + '/' + 'boundary.txt', directory + '/' + 'network_shape' + '/')
        shutil.copy2(directory + '/' + 'center_radius.txt', directory + '/' + 'network_shape' + '/')
        shutil.copy2(directory + '/' + 'kron_points.txt', directory + '/' + 'network_shape' + '/')
        shutil.copy2(directory + '/' + 'kronecker_hull.png', directory + '/' + 'network_shape' + '/')
        shutil.copy2(directory + '/' + 'kronecker_hull.fig', directory + '/' + 'network_shape' + '/')

        # zips network_shape directory
        shutil.make_archive(directory + 'network_shape', 'zip', directory + 'network_shape')


