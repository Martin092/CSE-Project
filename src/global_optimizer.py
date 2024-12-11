"""Global Optimizer module"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Literal
from mpi4py import MPI  # pylint: disable=E0611
from ase import Atoms
from ase.io import write, Trajectory
import numpy as np

from src.utility import Utility


class GlobalOptimizer(ABC):
    """
    Class Interface for Global Optimizers
    """

    def __init__(
        self,
        local_optimizer: Any,
        calculator: Any,
        comm: MPI.Intracomm | None = None,
    ) -> None:
        """
        Global Optimizer Class Constructor
        :param local_optimizer: ASE Optimizer.
        :param calculator: ASE Calculator.
        :param comm: MPI global communicator object.
        """
        self.local_optimizer: Any = local_optimizer
        self.current_iteration: int = 0
        self.calculator: Any = calculator
        self.utility: Utility | None = None
        self.execution_time: float = 0.0
        self.comm: MPI.Intracomm | None = comm
        self.current_cluster: Atoms | None = None
        self.best_potential: float = float("inf")
        self.best_config: Atoms | None = None
        self.potentials: List[float] = []
        self.configs: List[Atoms] = []
        self.num_atoms: int = 0
        self.atom_type: str = ""

    @abstractmethod
    def iteration(self) -> None:
        """
        Performs single iteration of the Global Optimizer algorithm.
        :return: None
        """

    @abstractmethod
    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :return: True if convergence criteria is met, otherwise False.
        """

    def setup(
        self,
        num_atoms: int,
        atom_type: str,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
    ) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided.
        :param num_atoms: Number of atoms in cluster.
        :param atom_type: Atomic type of cluster.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :return: None.
        """
        self.current_iteration = 0
        self.num_atoms = num_atoms
        self.atom_type = atom_type
        self.utility = Utility(self, num_atoms, atom_type)
        self.current_cluster = self.utility.generate_cluster(initial_configuration)

    def run(
        self,
        num_atoms: int,
        atom_type: str,
        max_iterations: int,
        seed: int | None = None,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
    ) -> None:
        """
        Executes the Global Optimizer algorithm.
        :param num_atoms: Number of atoms in cluster to optimize for.
        :param atom_type: Atomic type of cluster.
        :param max_iterations: Number of maximum iterations to perform.
        :param seed: Seeding for reproducibility.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :return: None.
        """
        np.random.seed(seed)
        start_time = time.time()
        self.setup(num_atoms, atom_type, initial_configuration)

        while self.current_iteration < max_iterations and not self.is_converged():
            self.iteration()
            self.current_iteration += 1

        self.execution_time = time.time() - start_time

    def write_configuration(self, filename: str) -> None:
        """
        Writes cluster to a .xyz file.
        :param filename: Name of file, without .xyz extension.
        :return: None, writes to file.
        """
        filename = filename if filename[-4:] == ".xyz" else filename + ".xyz"
        write(f"../data/optimizer/{filename}.xyz", self.best_config)  # type: ignore

    def write_trajectory(self, filename: str) -> None:
        """
        Writes all clusters in the history to a trajectory file
        :param filename: Name of the trajectory file, without .traj extension
        :return: None, writes to file
        """
        with Trajectory(filename, mode="w") as traj:  # type: ignore
            for cluster in self.configs:
                cluster.center()  # type: ignore
                traj.write(cluster)  # pylint: disable=E1101
