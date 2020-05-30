import json
import glob
import hashlib
import logging
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy.distutils.system_info as sysinfo
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse
import os

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR)

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k,v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = "_".join([str(self.features[node])]+sorted([str(deg) for deg in degs]))
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for iteration in range(self.iterations):
            self.features = self.do_a_recursion()
        
def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        features = {int(k):v for k,v, in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k):v for k,v, in list(features)}
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph,features,rounds)
    doc = TaggedDocument(words = machine.extracted_features , tags = ["g_" + name])
    return doc
        
def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".json")
        out.append([int(identifier)] + list(model.docvecs["g_"+identifier]))

    # print(out)
    out = pd.DataFrame(out,columns = ["type"] +["x_" +str(dimension) for dimension in range(dimensions)])
    out = out.sort_values(["type"])
    out.to_csv(output_path, index = None)

def run_graph2vec(args):
    """
    Main function to read the graph list, extract features, learn the embedding and save it.
    :param args: Object with the arguments.
    """
    directory = 'downloads' + '/' + args.id + '/'

    dimensions = 3
    workers = 4
    epochs = 10
    min_count = 5
    wl_iterations = 2
    learning_rate = 0.025
    down_sampling = 0.0001

    step = int(args.step)
    for p in range(step, 100, step):
        # print(p)
        graphs = glob.glob(directory + '/' + str(p) + "/*.json")
        output_path = directory + '/' + str(p) + "/g2v.csv"
        print("\nFeature extraction started.\n")
        document_collections = Parallel(n_jobs=int((len(os.sched_getaffinity(0)) / 2)))(delayed(feature_extractor)(g, wl_iterations) for g in tqdm(graphs))
        print("\nOptimization started.\n")

        model = Doc2Vec(document_collections,
                        size = dimensions,
                        window = 0,
                        min_count = min_count,
                        dm = 0,
                        sample = down_sampling,
                        workers = workers,
                        iter = epochs,
                        alpha = learning_rate)

        save_embedding(output_path, model, graphs, dimensions)

    graphs = glob.glob(directory + "/*.json")
    output_path = directory + "/g2v.csv"
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=int((len(os.sched_getaffinity(0)) / 2)))(
        delayed(feature_extractor)(g, wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    size=3,
                    window=0,
                    min_count=min_count,
                    dm=0,
                    sample=down_sampling,
                    workers=workers,
                    iter=epochs,
                    alpha=learning_rate)
    save_embedding(output_path, model, graphs, dimensions)

    output_path = directory + "/g2v_points.txt"
    points = []
    f = open(output_path, 'w')
    f.write('x1,x2,x3,sampling_proportion\n')
    for p in range(step, 100, step):
        data = pd.read_csv(directory + '/' + str(p) + "/g2v.csv")
        (row_num, col_num) = data.shape
        for i in range(0, row_num):
            f.write('{},{},{},{}\n'.format(str(data.ix[i, 'x_0']), str(data.ix[i, 'x_1']), str(data.ix[i, 'x_2']), str(p)))
            points.append([float(data.ix[i, 'x_0']), float(data.ix[i, 'x_1']), float(data.ix[i, 'x_2']), float(p)])
    data = pd.read_csv(directory + "/g2v.csv")
    f.write('{},{},{},{}\n'.format(str(data.ix[0, 'x_0']), str(data.ix[0, 'x_1']), str(data.ix[0, 'x_2']), str(100)))
    points.append([float(data.ix[0, 'x_0']), float(data.ix[0, 'x_1']), float(data.ix[0, 'x_2']), float(100)])
    f.close()
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-path",
                        nargs="?",
                        default="./dataset/",
                        help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./features/nci1.csv",
                        help="Embeddings path.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
                        help="Number of dimensions. Default is 128.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
                        help="Number of workers. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")

    parser.add_argument("--min-count",
                        type=int,
                        default=5,
                        help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations",
                        type=int,
                        default=2,
                        help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
                        help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.0001,
                        help="Down sampling rate for frequent features. Default is 0.0001.")

    parser.add_argument("--id",
                        default='test',
                        help="Down sampling rate for frequent features. Default is 0.0001.")
    print(parser.id)
    run_graph2vec(parser)