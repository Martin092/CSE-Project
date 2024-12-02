"""TODO: Write this."""

from typing import Any, List, Literal, Tuple
import time
import sys
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from sklearn.cluster import DBSCAN
from ase import Atoms, Atom
from ase.units import fs
from ase.md.langevin import Langevin
from ase.optimize.minimahopping import PassedMinimum
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from reference_code.rotation_matrices import rotation_matrix
from atom_parameters import getRcov_n


class Utility:
    """
    Class with all the methods to disturb a cluster
    """

    def __init__(self, global_optimizer: Any) -> None:
        self.global_optimizer = global_optimizer

    def random_step(self, cluster: Atoms) -> None:
        """
        Moves the highest energy atom in a random direction
        :param cluster: the cluster we want to disturb
        :return: result is written directly to cluster, nothing is returned
        """
        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)
        energy_before = cluster.get_potential_energy()  # type: ignore

        # random step from -1 to 1
        step_size = (np.random.rand(3) - 0.5) * 2

        attempts = 0
        while True:
            # every 100 attempts to find a new step, increase the step size by 1.
            # NOTE: probably not the best way to go about the algorithm not finding an appropriate step but works
            attempts += 1
            if attempts % 100 == 0:
                step_size += 1

            step = (np.random.rand(3) - 0.5) * 2 * step_size
            energy_before = self.global_optimizer.cluster_list[0].get_potential_energy()

            cluster.positions[index] += step
            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.box_length,
                self.global_optimizer.box_length,
            )

            energy_after = self.global_optimizer.cluster_list[0].get_potential_energy()

            # Metropolis criterion gives an acceptance probability based on temperature for each move
            if not self.metropolis_criterion(energy_before, energy_after, 1):
                cluster.positions[index] -= step
                continue
            break

    def metropolis_criterion(
        self, initial_energy: float, new_energy: float, temp: float = 0.8
    ) -> bool:
        """
        Metropolis acceptance criterion for accepting a new move based on temperature
        :param initial_energy: The energy of the cluster before the move
        :param new_energy: The energy of the cluster after the move
        :param temp: temperature at which we want the move to occur
        :return: whether the move is accepted
        """
        if (
            np.isnan(new_energy) or new_energy - initial_energy > 50
        ):  # Energy is way too high, bad move
            return False
        if new_energy > initial_energy:
            accept_prob = np.exp(-(new_energy - initial_energy) / temp)
            # We accept each move with a probability given by the Metropolis criterion
            return bool(np.random.rand() < accept_prob)
        # We went downhill, cool
        return True

    def check_atom_position(self, cluster: Atoms, atom: Atom) -> bool:
        """
        TODO: Write this.
        :param cluster:
        :param atom:
        :return:
        """
        if np.linalg.norm(atom.position) > self.global_optimizer.box_length:
            return False
        for other_atom in cluster:
            if (
                np.linalg.norm(atom.position - other_atom.position)
                < 0.5 * self.global_optimizer.covalent_radius
            ):
                return False
        return True

    def check_group_position(
        self, group_static: List[Atom], group_moved: List[Atom]
    ) -> bool:
        """
        TODO: Write this.
        """
        for atom in group_moved:
            if np.linalg.norm(atom.position) > self.global_optimizer.box_length:
                return False
            for other_atom in group_static:
                if (
                    np.linalg.norm(atom.position - other_atom.position)
                    < 0.5 * self.global_optimizer.covalent_radius
                ):
                    return False
        return True

    def angular_movement(self, cluster: Atoms) -> None:
        """
        Perform a rotational movement for the atom with the highest energy.
        :param cluster: The atomic cluster to modify
        """

        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)

        initial_positions = cluster.positions.copy()
        initial_energy = cluster.get_potential_energy()  # type: ignore
        max_attempts = 500
        temperature = 1.0

        cluster.set_center_of_mass([0, 0, 0])  # type: ignore

        for _ in range(max_attempts):
            vector = np.random.rand(3) - 0.5
            angle = np.random.uniform(0, 2 * np.pi)

            rotation = rotation_matrix(vector, angle)

            rotated_position = np.dot(rotation, cluster.positions[index])
            cluster.positions[index] = rotated_position

            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.box_length,
                self.global_optimizer.box_length,
            )

            new_energy = cluster.get_potential_energy()  # type: ignore

            if self.metropolis_criterion(initial_energy, new_energy, temperature):
                break
            cluster.positions = initial_positions
        else:
            print("WARNING: Unable to find a valid rotational move.", file=sys.stderr)

    def md(
        self,
        cluster: Atoms,
        temperature: float,
        mdmin: int,
        seed: int = int(time.time()),
    ) -> None:
        """
        Perform a Molecular Dynamics run using Langevin Dynamics
        :param cluster: Cluster of atoms
        :param temperature: Temperature in Kelvin
        :param mdmin: Number of minima to be found before MD run halts.
        Alternatively it will halt once we reach 10000 iterations
        :param seed: seed for random generation, can be used for testing
        """
        dyn = Langevin(
            cluster,
            timestep=5.0 * fs,  # Feel free to mess with this parameter
            temperature_K=temperature,
            friction=0.5 / fs,  # Feel free to mess with this parameter
            rng=np.random.default_rng(seed),
        )

        MaxwellBoltzmannDistribution(cluster, temperature_K=temperature)
        passed_minimum = PassedMinimum()  # type: ignore
        mincount = 0
        energies, oldpositions = [], []
        i = 0
        while mincount < mdmin and i < 10000:
            dyn.run(1)  # type: ignore # Run MD for 1 step
            energies.append(cluster.get_potential_energy())  # type: ignore
            passedmin = passed_minimum(energies)
            if passedmin:  # Check if we have passed a minimum
                mincount += 1  # Add this minimum to our mincount
            oldpositions.append(cluster.positions.copy())
            i += 1
        print("Number of MD steps: " + str(i))
        cluster.positions = oldpositions[passedmin[0]]
        cluster.positions = np.clip(
            cluster.positions,
            -self.global_optimizer.box_length,
            self.global_optimizer.box_length,
        )

    def twist(self, cluster: Atoms) -> Atoms:
        """
        TODO: Write this.
        :param cluster:
        :return:
        """
        # Twist doesn't have a check since it is a rotation, and it wouldn't collide with already existing atoms.
        group1, group2, normal = self.split_cluster(cluster)
        choice = np.random.choice([0, 1])
        chosen_group = group1 if choice == 0 else group2

        angle = np.random.uniform(0, 2 * np.pi)
        matrix = rotation_matrix(normal, angle)

        for atom in chosen_group:
            atom.position = np.dot(matrix, atom.position)

        return cluster

    def etching(self, cluster: Atoms) -> None:
        """
        TODO: Write this.
        :param cluster:
        :return:
        """

    def split_cluster(
        self,
        cluster: Atoms,
        p1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p3: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
    ) -> Tuple[
        List[Atom], List[Atom], np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]
    ]:
        """
        TODO: Write this.
        :param cluster:
        :param p1:
        :param p2:
        :param p3:
        :return:
        """
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        d = -np.dot(normal, p1)
        group1 = []
        group2 = []
        for atom in cluster:
            val = np.dot(normal, atom.position) + d
            if val > 0:
                group1.append(atom)
            else:
                group2.append(atom)
        return group1, group2, normal

    def align_cluster(self, cluster: Atoms) -> Atoms:
        """
        TODO: Write this.
        :param cluster:
        :return:
        """
        cl = np.array(cluster.positions)
        center_of_mass = np.mean(cl, axis=0)
        cluster_centered = cl - center_of_mass
        pca = PCA(n_components=3)
        pca.fit(cluster_centered)
        principal_axes = pca.components_
        rotated_cluster = np.dot(cluster_centered, principal_axes.T)
        cluster.positions = rotated_cluster
        return cluster

    def compare_clusters(self, cluster1: Atoms, cluster2: Atoms) -> np.bool:
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: boolean
        """
        return np.isclose(cluster1.get_potential_energy(), cluster2.get_potential_energy())  # type: ignore

    # Extracted from: https://github.com/ElsevierSoftwareX/SOFTX-D-23-00623/blob/main/minimahopping/md/dbscan.py#L24
    def get_eps(self, elements):
        rcovs = []
        for element in elements:
            rcovs.append(getRcov_n(element))
        rcovs = np.array(rcovs)
        rcov_mean = np.mean(rcovs)
        eps = 2.0 * rcov_mean
        return eps

    def get_com(self, positions, mass):
        total_mass = np.sum(mass)
        mass_3d = np.vstack([mass] * 3).T
        weighted_positions = positions * mass_3d
        com = np.sum(weighted_positions, axis=0)
        com /= total_mass
        return com

    def dbscan(
            self,
            eps,
            positions,
    ):
        db = DBSCAN(eps=eps, min_samples=2).fit(positions)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
#        assert n_noise == 0, "Some atoms in DBSCAN were recognized as noise"
        return labels, n_clusters

    def adjust_velocities(self, cluster, positions, velocities, elements, masses):
        com = self.get_com(positions, masses)
        eps = self.get_eps(elements)
        mass_3d = np.vstack([masses] * 3).T
        _e_kin = 0.5 * np.sum(mass_3d * velocities * velocities)
        _v_average = np.sqrt((2.0 * _e_kin) / (np.mean(masses) * velocities.shape[0]))
        shifts = np.zeros(positions.shape)
        labels, n_clusters = self.dbscan(eps, positions)

        print("Number of clusters: " + str(n_clusters))
        while n_clusters > 1:
            print("Number of clusters is still: " + str(n_clusters))
            for i in range(n_clusters):
                indices = np.where(labels == i)
                cluster_pos = positions[indices, :][0]
                cluster_mass = masses[indices]
                com_cluster = self.get_com(cluster_pos, cluster_mass)
                shift = (com - com_cluster) / np.linalg.norm(com - com_cluster)
                indices = np.where(labels == i)
                shifts[indices, :] = shift

            cluster.set_positions(cluster.get_positions() + shifts)
            positions = cluster.get_positions()
            labels, n_clusters = self.dbscan(eps, positions)

    def fix_fragmentation(self, cluster: Atoms):
        """
        Move every atom in the cluster towards the center of mass
        :param cluster: Cluster of atoms
        :return: None, cluster is edited by reference
        """
        self.adjust_velocities(
            cluster,
            cluster.get_positions(),
            cluster.get_velocities(),
            cluster.get_atomic_numbers(),
            cluster.get_masses(),
        )

