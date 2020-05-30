'''
Reference implementation of Network Shapes.
Author: Shengmin Jin
For more details, refer to the paper:
Representing Networks with 3D Shapes
Shengmin Jin and Reza Zafarani
IEEE International Conference on Data Mining (ICDM), 2018

Versions: 1.0    The Original Version
          1.1    Code Optimization
          1.2    Bug fix for python subprocess hang
          1.3    Bug fix for wrong output
          2.0    Added different sampling methods
          2.1    Added graph2vec as an embedding method
          2.2    Added 
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

import matlab.engine
from tqdm import tqdm
from joblib import Parallel, delayed
from Sampler import Sampler
from Fitter import Fitter
from Embedder import Embedder



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generating a 3D Network Shapes with a Kronecker Hull')

    parser.add_argument('-name', required=False,
                         default='as20graph',
                         help='Network name')

    parser.add_argument('-file', required=False,
                        default='as20graph.txt',
                        help='Input edge list file name.')

    parser.add_argument('-step', required=False,
                        default=10,
                        help='Sampling proportion step (percentage)')

    parser.add_argument('-t', required=False,
                        default=10,
                        help='Number of samples for each sampling proporation"')

    parser.add_argument('-embedding', required=False,
                        default='graph2vec',   # or kron
                        help='Embedding Methods')

    parser.add_argument('-sampling', required=False,
                        default='randomEdge',
                        help='Sampling Methods')

    parser.add_argument('-fitting', required=False,
                        default='convexhull',
                        help='Fitting Methods')

    parser.add_argument('-z', '--zip', required=False,
                        help='Copy and Zip certain files to a new directory for downloading',
                        action='store_true')


    args = parser.parse_args()
    uid = args.id
    step = int(args.step)
    nos = int(args.t)
    embedding_method = args.embedding
    sampling_method = args.sampling
    fitting_method = args.fitting



    network_name = 'downloads' + '/' + uid
    directory = network_name + '/'
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    print("Network name: {}".format(network_name))
    print("Input edge file: {}".format(args.file))
    print("Sampling proportion step: {}".format(args.step))
    print("Number of samples for each sampling proportion: {}".format(args.t))
    print("Embedding Method: {}".format(args.embedding))
    print("Sampling Method: {}".format(args.sampling))
    print("Fitting Method: {}".format(args.fitting))


    sampler = Sampler(args)
    sampler.sample()
    embedder = Embedder(args)
    points = embedder.embed()
    fitter = Fitter(args)
    fitter.fit(points)
