from abc import ABC, abstractmethod
from ase import Atoms
import numpy as np
from Disturber import Disturber
from ase.io import write

class GlobalOptimizer(ABC):

    def __init__(self, num_clusters: int, localOptimizer, atoms: int, atom_type: str, calculator):
        self.history = []
        self.clusterList = []
        self.optimizers = []
        self.localOptimizer = localOptimizer
        self.currentIteration = 0
        self.atoms = atoms
        self.covalentRadius = 1.0
        self.boxLength = 2 * self.covalentRadius * (1/2 + ((3.0 * self.atoms) / ( 4 * np.pi * np.sqrt(2)))**(1/3))
        self.atom_type = atom_type
        self.calculator = calculator
        self.disturber = Disturber(self.localOptimizer, self)

        for i in range(num_clusters):
            positions = ( np.random.rand(self.atoms, 3) - 0.5 ) *  self.boxLength * 1.5 #1.5 is a magic number
            # In the future, instead of number of atoms, we ask the user to choose how many atoms they want for each atom type.
            clus = Atoms(self.atom_type + str(self.atoms), positions=positions)
            clus.calc = calculator()
            self.clusterList.append(clus)
            self.history.append([clus.copy()])
            opt = localOptimizer(clus, logfile='log.txt')
            self.optimizers.append(opt)

    @abstractmethod
    def iteration(self):
        pass

    @abstractmethod
    def isConverged(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    def run(self, maxIterations):
        self.setup()

        while self.currentIteration < maxIterations and not self.isConverged():
            print(self.currentIteration)
            self.iteration()
            self.currentIteration += 1

    def write_to_file(self, filename: str, cluster_index=0):
        """
        Writes the cluster to a .xyz file.
        :param filename: the name of the file, does not matter if it has the .xyz extension
        :param cluster_index: which cluster will be written
        """
        filename = filename if filename[-4:] == ".xyz" else filename + ".xyz"
        write(f'clusters/{filename}', self.clusterList[cluster_index])

    def append_history(self):
        """
        Appends copies of all the clusters in the clusterList to the history.
        Copies are used since clusters are passed by reference
        :return:
        """
        for i, cluster in enumerate(self.clusterList):
            self.history[i].append(cluster.copy())

    @staticmethod
    def compare_clusters(cluster1, cluster2):
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: boolean
        """
        return np.isclose(cluster1.get_potential_energy(), cluster2.get_potential_energy())
