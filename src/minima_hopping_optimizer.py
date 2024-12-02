"""
Implementation of the Minima Hopping Algorithm
"""

from typing import List, Any
import numpy as np
from scipy.spatial.distance import pdist

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.io import write
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

from src.global_optimizer import GlobalOptimizer


class MinimaHoppingOptimizer(GlobalOptimizer):
    """
    Implementation of the Minima Hopping Optimizer
    """

    def __init__(
        self,
        num_clusters: int,
        atoms: int,
        atom_type: str,
        temperature: float,
        local_optimizer: Any = BFGS,
        calculator: Any = LennardJones,
    ):
        super().__init__(num_clusters, local_optimizer, atoms, atom_type, calculator)
        self.alpha_r = 1.02
        self.alpha_a = 1 / 1.02
        self.beta_s = 1.05
        self.beta_o = 1.05
        self.beta_n = 1 / 1.05
        self.temperature = temperature
        self.e_diff = 0.5
        self.mdmin = 3
        self.m_cur = [None] * num_clusters
        self.minima_history: List[Atoms] = []
        self.last_energy = float("inf")
        self.box_length *= 0.75

    def iteration(self) -> None:
        """
        Runs a single iteration of Minima hopping
        :return:
        """
        for cluster_index, cluster in enumerate(self.cluster_list):
            self.last_energy = cluster.get_potential_energy()
            self.utility.md(cluster, self.temperature, self.mdmin)
            # cluster.set_momenta(np.zeros(cluster.get_momenta().shape))

            with self.local_optimizer(cluster, logfile="log.txt") as opt:
                opt.run(fmax=0.2)

            self.check_results(cluster, cluster_index)

            # If the maximum distance between 2 atoms is greater than some number, then we should check
            # if we have fragmentation

            distances = pdist(cluster.get_positions())
            if (
                np.max(distances) > 2 * np.sqrt(3) * self.box_length * 0.5
            ):  # fuck around with this
                print(np.max(distances))
                print(2 * np.sqrt(3) * self.box_length * 0.5)
                print("Checking for fragmentation")
                self.utility.fix_fragmentation(cluster)

            if self.temperature > 3000: #Force temperature back down if it goes too high
                self.temperature = 1000

            self.append_history()
            print("Temperature: " + str(self.temperature))
            print("Energy: " + str(cluster.get_potential_energy()))
            print()

    def check_results(self, m: Atoms, cluster_index: int) -> None:
        """
        Checks the outcome of a minima hopping run and changes temperature and e_diff variables depending on the outcome
        :param m: Minima we found
        :param cluster_index: Current index in the cluster list
        :return: None
        """
        if (
            self.m_cur[cluster_index] is not None
        ):  # Case 1: We did not find a new minimum
            if self.utility.compare_clusters(self.m_cur[cluster_index], m):  # type: ignore
                # print(self.m_cur[i].get_potential_energy())
                # print(m.get_potential_energy())
                # print("2 minima are the same")
                self.temperature *= self.beta_s
                return

            if (
                m.get_potential_energy() - self.m_cur[cluster_index].get_potential_energy()  # type: ignore
                < self.e_diff
            ):  # Check if energy has decreased enough to be considered a new minimum
                # print("Energy between 2 minima has decreased enough")
                self.e_diff *= self.alpha_a
                self.m_cur[cluster_index] = m.copy()  # type: ignore
                self.m_cur[cluster_index].calc = self.calculator()  # type: ignore
            else:
                self.e_diff *= self.alpha_r
        else:
            self.m_cur[cluster_index] = m.copy()  # type: ignore
            self.m_cur[cluster_index].calc = self.calculator()  # type: ignore

        for (
            minima
        ) in (
            self.minima_history
        ):  # Is this a minima we've seen before? Change temperature accordingly
            if self.utility.compare_clusters(m, minima):
                # print(minima.get_potential_energy())
                # print(m.get_potential_energy())
                # print("We've seen this minima before")
                self.temperature *= self.beta_o
                return

        # print("We've never seen this minima before")
        self.minima_history.append(m.copy())  # type: ignore
        self.minima_history[-1].calc = self.calculator()
        self.temperature *= self.beta_n

    def is_converged(self) -> bool:
        return False


mh = MinimaHoppingOptimizer(
    num_clusters=1,
    atoms=19,
    atom_type="Fe",
    temperature=100,
)
mh.run(500)
best_cluster = mh.get_best_cluster_found()
mh.write_trajectory("../clusters/minima_progress.traj")
print("Best energy found: ")
print(best_cluster.get_potential_energy())
write("../clusters/minima_optimized.xyz", best_cluster)

traj = TrajectoryReader("../clusters/minima_progress.traj")  # type: ignore
BEST_INDEX = 0
for i, _ in enumerate(traj):
    if mh.utility.compare_clusters(traj[i], best_cluster):
        print("Found best cluster at iteration: ")
        print(i)
        BEST_INDEX = i
        break
view(traj[: BEST_INDEX + 1])  # type: ignore
