import random, os
import networkx as nx

class Sampler:
    def __init__(self, args):
        self.args = args
        self.network_name = args.name
        self.edgelist = args.file
        self.G = nx.read_edgelist(self.edgelist, delimiter='\t')
        self.nodes = list(self.G.nodes())
        self.step = int(args.step)
        self.nos = int(args.t)
        self.sampling_method = args.sampling
        self.embedding_method = args.embedding
        self.directory = self.network_name + '/'


    def sample(self):
        # if self.embedding_method == 'kroneckerPoint':
        nx.write_edgelist(self.G.to_directed(), self.directory + '100.edgelist', delimiter='\t', data=False)
        #elif self.embedding_method == 'graph2vec':
        json_path = self.directory + '100.json'
        self.write_json(json_path, self.G.to_directed())
        # get sample graphs
        for p in range(self.step, 100, self.step):
            print('Sampling ' + str(p) + '% subgraphs')
            os.mkdir(self.directory + str(p) + '/')
            for i in range(0, self.nos):
                if self.sampling_method == 'randomEdge':
                    self.random_edge_sampling(self.directory, p, i)
                elif self.sampling_method == 'randomNode':
                    self.random_node_sampling(self.directory, p, i)
                elif self.sampling_method == 'randomWalk':
                    self.random_walk_with_restart_sampling(self.directory, p, i)

    # sample a subgraph
    def random_node_sampling(self, directory, p, i):
        random.shuffle(self.nodes)
        size = int(len(self.nodes) * float(p) / 100)
        sample = self.nodes[:size]
        sub_g = self.G.subgraph(sample)
        sub_g = sub_g.to_directed()  # Forgot to add
        self.write(sub_g, directory, p, i)

    def random_edge_sampling(self, directory, p, i):
        sub_g = self.G.copy()
        edges = [e for e in sub_g.edges]
        random.shuffle(edges)
        size = int(len(edges) * float(100 - p) / 100)
        sub_g.remove_edges_from(edges[:size])
        sub_g = sub_g.to_directed()  # Forgot to add
        self.write(sub_g, directory, p, i)

    def random_walk_with_restart_sampling(self, directory, p, i, restart_prob=0.15, jump_iteration=10, seed=None):
        # set random seed
        random.seed(seed)

        # sample size round down to interger
        sample_size = int(len(self.nodes) * float(p) / 100)

        # set starting node
        startnode = random.choice(self.nodes)
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
                nextnode = random.choice(list(self.G[currentnode]))
                currentnode = nextnode
            # find a new startnode if number of nodes in sample does not grow
            if restart_iteration < jump_iteration:
                restart_iteration += 1
            else:
                if last_number_of_nodes == len(nodelist):
                    startnode = random.choice(self.nodes)
                    currentnode = startnode
                restart_iteration = 0
                last_number_of_nodes = len(nodelist)
        sub_g = self.G.subgraph(nodelist).to_directed()
        self.write(sub_g, directory, p, i)

    def write(self, sub_g, directory, p, i):
        # if self.embedding_method == 'kroneckerPoint':
        nx.write_edgelist(sub_g, directory + str(p) + '/'
                              + str(i) + '.edgelist', delimiter='\t', data=False)
        # elif self.embedding_method == 'graph2vec':
        json_path = directory + str(p) + '/' + str(i) + '.json'
        self.write_json(json_path, sub_g)

    def write_json(self, json_path, graph):
        with open(json_path, 'w') as output_file:
            output_file.write('{"edges": [')
            for index, edge in enumerate(graph.edges()):
                n1, n2 = edge
                if index < len(graph.edges()) - 1:
                    output_file.write('[{}, {}],'.format(n1, n2))
                else:
                    output_file.write('[{}, {}]'.format(n1, n2))
            output_file.write(']}')
