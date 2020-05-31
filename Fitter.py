import os, shutil
import matlab.engine
import glob

class Fitter:
    def __init__(self, args):
        self.args = args
        self.network_name = args.name
        self.step = int(args.step)
        self.nos = int(args.t)
        self.sampling_method = args.sampling
        self.embedding_method = args.embedding
        self.fitting_method = args.fitting
        self.directory = self.network_name + '/' + self.embedding_method + '/'
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        os.mkdir(self.directory)
        self.zip = args.zip


    def fit(self, points):

        # if self.fitting_method == 'convexhull':
        self.create_kronecker_hull(self.directory, points, self.network_name)
        # elif self.fitting_method == 'cuboid':
        self.create_cuboid(self.directory, points, self.network_name)
        # elif self.fitting_method == 'sphere':
        self.create_sphere(self.directory, points, self.network_name)

        if self.zip:
            # makes new directory network_shape and copies them to it
            os.mkdir(self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '/' + 'boundary.txt', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '/' + 'center_radius.txt', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '/' + 'kron_points.txt', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '/' + 'kronecker_hull.png', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '/' + 'kronecker_hull.fig', self.directory + '/' + 'network_shape' + '/')
            dest_dir = self.directory + '/' + 'network_shape' + '/'
            for file in glob.glob(self.directory + '*.txt'):
                shutil.copy(file, dest_dir)
            for file in glob.glob(self.directory + '*.png'):
                shutil.copy(file, dest_dir)
            for file in glob.glob(self.directory + '*.fig'):
                shutil.copy(file, dest_dir)
            # shutil.copy2(self.directory + '*.txt', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '*.png', self.directory + '/' + 'network_shape' + '/')
            # shutil.copy2(self.directory + '*.fig', self.directory + '/' + 'network_shape' + '/')

            # zips network_shape directory
            shutil.make_archive(self.directory + 'network_shape', 'zip', self.directory + 'network_shape')

    def create_kronecker_hull(self, directory, points, display_name):
        eng = matlab.engine.start_matlab()
        eng.get_convex_hull(matlab.double(points), directory, display_name)

    def create_cuboid(self, directory, points, display_name):
        eng = matlab.engine.start_matlab()
        eng.get_cuboid(matlab.double(points), directory, display_name)

    def create_sphere(self, directory, points, display_name):
        eng = matlab.engine.start_matlab()
        eng.get_sphere(matlab.double(points), directory, display_name)
