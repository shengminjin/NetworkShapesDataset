import os
from subprocess import PIPE
from graph2vec import run_graph2vec
import subprocess
from tqdm import tqdm
import re
import glob
from joblib import Parallel, delayed

class Embedder:
    def __init__(self, args):
        self.args = args
        self.network_name = args.name
        self.step = int(args.step)
        self.nos = int(args.t)
        self.sampling_method = args.sampling
        self.embedding_method = args.embedding
        self.directory = self.network_name + '/'


    def embed(self):
        if self.embedding_method == 'kroneckerPoint':
            kronfit_jobs = []
            input_file = self.directory + '100.edgelist'
            output_file = self.directory + '100_output.dat'
            kronfit_jobs.append((input_file, output_file))
            for p in range(self.step, 100, self.step):
                for i in range(0, self.nos):
                    input_file = self.directory + str(p) + '/' + str(i) + '.edgelist'
                    output_file = self.directory + str(p) + '/' + str(i) + '_output.dat'
                    kronfit_jobs.append((input_file, output_file))
            print("Running Kronfit for each graph")
            Parallel(n_jobs=int((len(os.sched_getaffinity(0)) / 2)))(
                delayed(self.kronfit)(kronfit_job)
                for kronfit_job in tqdm(kronfit_jobs))
            print("Kronfit Finished")

            output = open(self.directory + 'kron_points.txt', 'w')
            points = []
            for p in range(self.step, 100, self.step):
                for i in range(0, self.nos):
                    output_file = self.directory + str(p) + '/' + str(i) + '_output.dat'
                    (a, b, d) = self.read_kron_point(output_file)
                    output.write(str(a) + ',' + str(b) + ',' + str(d) + ',' + str(p) + '\n')
                    points.append([float(a), float(b), float(d), float(p)])

            full_output_file = self.directory + '100_output.dat'
            (a, b, d) = self.read_kron_point(full_output_file)
            output.write(str(a) + ',' + str(b) + ',' + str(d) + ',' + str(100) + '\n')
            points.append([float(a), float(b), float(d), float(100)])
            output.close()
            return points

        elif self.embedding_method == 'graph2vec':
            return run_graph2vec(self.args)

    # def kronfit(self, kronfit_job):
    #     #     input_file_path = kronfit_job[0]
    #     #     output_file_path = kronfit_job[1]
    #     #     # print(input_file_path, output_file_path)
    #     #
    #     #     if not os.path.exists(output_file_path):
    #     #         cmd = 'kronfit', '-i:' + input_file_path, '-n0:2', '-gi:20', '-o:' + output_file_path
    #     #         subprocess.Popen(cmd, stdout=PIPE).communicate()
    def kronfit(self, kronfit_job):
        input_file_path = kronfit_job[0]
        output_file_path = kronfit_job[1]
        if not os.path.exists(output_file_path):
            cmd = 'kronfit', '-i:' + input_file_path, '-n0:2', '-gi:20', '-o:' + output_file_path
            subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()

    def read_kron_point(self, output_file):
        with open(output_file, 'r') as myfile:
            s = myfile.read()
            ret = re.findall(r'\[([^]]*)\]', s)
            split = ret[0].split(',')
            a = split[0]
            b = split[1].split(';')[0].strip()
            d = split[2].strip()

        return (a, b, d)
