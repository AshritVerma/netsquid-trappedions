import unittest
from abc import abstractmethod
from copy import deepcopy

import math
import matplotlib.pyplot as plt
import numpy as np

import netsquid as ns
from netsquid import BellIndex
from netsquid.components.instructions import INSTR_INIT
from netsquid.qubits.operators import H
from netsquid_trappedions.instructions import INSTR_INIT_BELL
from netsquid_trappedions.ion_trap import IonTrap
from netsquid_trappedions.programs import ion_trap_one_qubit_hadamard

ns.qubits.qformalism.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)


class SampleDensityMatrix:
    """Base class for sim that investigate the effects of collective dephasing."""

    def __init__(self, total_time, num_intervals):

        self.num_intervals = num_intervals + 1
        self.density_matrices_sample_x_basis = [[] for _ in range(self.num_intervals)]
        self.density_matrices_average_x_basis = [None] * self.num_intervals
        self.density_matrices_error_x_basis = [None] * self.num_intervals
        self.times = [i * total_time / num_intervals for i in range(self.num_intervals)]
        self.ion_trap = None

    def average_dm(self):

        for time_index in range(self.num_intervals):
            self.density_matrices_average_x_basis[time_index] = np.mean(
                self.density_matrices_sample_x_basis[time_index], axis=0)
            density_matrices_standard_deviation_x_basis = np.std(
                self.density_matrices_sample_x_basis[time_index], axis=0)
            num_results = len(self.density_matrices_sample_x_basis[time_index])
            self.density_matrices_error_x_basis[time_index] = \
                density_matrices_standard_deviation_x_basis / np.sqrt(num_results)

    @abstractmethod
    def prepare_ion_trap(self):
        pass

    @abstractmethod
    def convert_to_x_basis(self, density_matrix):

        density_matrix_x_basis = None
        return density_matrix_x_basis

    def log_dm(self, time_index):

        qubit = self.ion_trap.peek([0])[0]
        density_matrix = qubit.qstate.dm
        density_matrix_x_basis = self.convert_to_x_basis(density_matrix)
        self.density_matrices_sample_x_basis[time_index].append(deepcopy(density_matrix_x_basis))

    def single_run(self):

        self.ion_trap.reset()
        self.prepare_ion_trap()
        self.log_dm(0)
        for time_index in range(1, self.num_intervals):
            duration = self.times[time_index] - self.times[time_index - 1]
            ns.sim_run(duration=duration)
            self.log_dm(time_index=time_index)

    def simulate(self, num_runs):

        for _ in range(num_runs):
            self.single_run()
        self.average_dm()

    def dm_key_figures(self, dm_x_basis, dm_error_x_basis=None):

        dm_pp = [np.abs(dm_x_basis[time_index][0][0]) for time_index in range(self.num_intervals)]
        dm_pm = [np.abs(dm_x_basis[time_index][0][1]) for time_index in range(self.num_intervals)]
        dm_pp_error = None
        dm_pm_error = None
        if dm_error_x_basis is not None:
            dm_pp_error = [np.abs(dm_error_x_basis[time_index][0][0]) for time_index in range(self.num_intervals)]
            dm_pm_error = [np.abs(dm_error_x_basis[time_index][0][1]) for time_index in range(self.num_intervals)]
        return dm_pp, dm_pp_error, dm_pm, dm_pm_error

    def plot(self, dm_x_basis, dm_error_x_basis=None, name='', onlyplusplus=True, pluspluscolor='red'):

        # TODO: can all commented code be removed?
        dm_pp, dm_pp_error, dm_pm, _ = self.dm_key_figures(dm_x_basis, dm_error_x_basis)
        plottimes = [t * 1e-7 for t in self.times]
        plt.errorbar(x=plottimes, y=dm_pp, yerr=dm_pp_error, fmt='-', color=pluspluscolor, label=(name + ' ++'))
        if onlyplusplus is False:
            plt.plot(plottimes, dm_pm, '--', color='violet', label=(name + ' +-'))

            # code below can be used to also plot other matrix elements, but these are not independent from ++ and +-
            # dm_mm = [np.abs(dm_x_basis[time_index][1][1]) for time_index in range(self.num_intervals)]
            # dm_mp = [np.abs(dm_x_basis[time_index][1][0]) for time_index in range(self.num_intervals)]
            # plt.plot(plottimes, dm_mm, '-', color='green', label=(name + ' --'))
            # plt.plot(plottimes, dm_mp, '--', color='crimson', label=(name + ' -+'))
        plt.xlabel('time (ms)')
        plt.ylabel('absolute value of density matrix coefficent')
        plt.xlim(left=0, right=plottimes[-1])
        plt.ylim(bottom=0, top=1.01)
        if onlyplusplus is True:
            plt.ylim(bottom=.5)
        plt.legend()

    def results_plot(self, onlyplusplus=True, error=True, name='', pluspluscolor='red'):

        dm_error_x_basis = None if error is False \
            else self.density_matrices_error_x_basis
        self.plot(dm_x_basis=self.density_matrices_average_x_basis,
                  dm_error_x_basis=dm_error_x_basis,
                  onlyplusplus=onlyplusplus,
                  name=name,
                  pluspluscolor=pluspluscolor)


class OneQubitDephasing(SampleDensityMatrix):
    """Check collective dephasing of one qubit.
    Taking a large enough sample should reveal Gaussian dephasing of the density matrix."""

    def __init__(self, total_time, num_intervals):
        super().__init__(total_time, num_intervals)
        self.ion_trap = IonTrap(num_positions=1, coherence_time=1E7)
        plt.title("density matrix for single qubit in ion trap")

    def prepare_ion_trap(self):
        self.ion_trap.execute_instruction(instruction=INSTR_INIT,
                                          qubit_mapping=[0])
        ns.sim_run()
        self.ion_trap.execute_program(ion_trap_one_qubit_hadamard)
        ns.sim_run()

    def convert_to_x_basis(self, density_matrix):
        density_matrix_x_basis = H.arr @ density_matrix @ H.arr
        return density_matrix_x_basis

    def expected_gaussian_dephasing(self):
        dm_x_basis = [[] for _ in range(self.num_intervals)]
        coherence_time = self.ion_trap.properties["coherence_time"]
        for time_index in range(self.num_intervals):
            q = (1 + np.exp(-2 * self.times[time_index] ** 2 / coherence_time ** 2)) / 2
            dm_x_basis[time_index] = np.array([[q, 0], [0, 1 - q]])

        return dm_x_basis

    def expected_gaussian_dephasing_plot(self, onlyplusplus=True, name=''):
        self.plot(dm_x_basis=self.expected_gaussian_dephasing(), onlyplusplus=onlyplusplus, pluspluscolor='blue',
                  name=name)

    def compare_results_to_theory_plot(self, error=True):
        self.results_plot(error=error, name='simulation')
        self.expected_gaussian_dephasing_plot(name='expectation')
        # plt.show()


class DephasingFreeSubspace(SampleDensityMatrix):
    """Check the effect of collective dephasing on a dephasing-free subspace."""

    def __init__(self, total_time, num_intervals):
        super().__init__(total_time, num_intervals)
        self.ion_trap = IonTrap(num_positions=2)
        self.ion_trap.add_instruction(INSTR_INIT_BELL, duration=0)
        plt.title('qubits encoded in dephasing-free subspace')
        self.conversion_matrix()

    def conversion_matrix(self):
        plusvector = np.array([0, 1, 1, 0]) / math.sqrt(2)
        minvector = np.array([0, 1, -1, 0]) / math.sqrt(2)
        zerovector = np.array([1, 0])  # , 0, 0])
        onevector = np.array([0, 1])  # , 0, 0])
        self.conversion_matrix = np.outer(zerovector, plusvector) + np.outer(onevector, minvector)

    def prepare_ion_trap(self):
        self.ion_trap.execute_instruction(INSTR_INIT_BELL, [0, 1], bell_index=BellIndex(1), bool_physical=False)
        ns.sim_run()

    def log_dm(self, time_index):
        self.ion_trap.peek([1])
        super().log_dm(time_index=time_index)

    def convert_to_x_basis(self, density_matrix):
        density_matrix_x_basis = self.conversion_matrix @ density_matrix @ self.conversion_matrix.conj().T

        return density_matrix_x_basis


class TestOneQubitCollectiveDephasing(unittest.TestCase):

    def setUp(self) -> None:
        ns.sim_reset()

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_simulation_against_analytical_model(self):
        num_intervals = 5
        num_runs = 100
        total_time = 2e7
        self.one_qubit_dephasing = OneQubitDephasing(total_time=total_time, num_intervals=num_intervals)
        self.one_qubit_dephasing.simulate(num_runs=num_runs)
        expected_density_matrix_x_basis = self.one_qubit_dephasing.expected_gaussian_dephasing()
        expected_pp, _, expected_pm, _ = self.one_qubit_dephasing.dm_key_figures(expected_density_matrix_x_basis)
        simulated_pp, error_pp, simulated_pm, error_pm = \
            self.one_qubit_dephasing.dm_key_figures(self.one_qubit_dephasing.density_matrices_average_x_basis,
                                                    self.one_qubit_dephasing.density_matrices_error_x_basis)
        for t in range(1, num_intervals + 1):
            # check whether expected and simulated density matrix elements (diagonal and off diagonal)
            # agree within 3 standard deviations of the mean (confidence interval 99.73 %)
            self.assertAlmostEqual(expected_pp[t], simulated_pp[t], delta=3 * error_pp[t])
            self.assertAlmostEqual(expected_pm[t], simulated_pm[t], delta=3 * error_pm[t])


class TestDephasingFreeSubspace(unittest.TestCase):

    def setUp(self) -> None:
        ns.sim_reset()

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_dephasing_free_subspace(self):
        num_intervals = 5
        num_runs = 1  # we expect perfect agreement, so we don't need statistics
        total_time = 2e7
        self.dephasing_free_subspace = DephasingFreeSubspace(total_time=total_time, num_intervals=num_intervals)
        self.dephasing_free_subspace.simulate(num_runs=num_runs)
        reference_density_matrix = [[1, 0], [0, 0]]
        self.assertTrue(np.all(np.isclose(self.dephasing_free_subspace.density_matrices_average_x_basis,
                                          reference_density_matrix, atol=0.000001)))


if __name__ == "__main__":
    # The code below can be used to generate plots.

    # Uncomment below to plot the evolution of a qubit encoded in decoherence free subspace.

    # dephasing_free_subspace = DephasingFreeSubspace(total_time=2e7, num_intervals=20)
    # dephasing_free_subspace.simulate(num_runs=1)
    # dephasing_free_subspace.results_plot(name='dephasing free subspace', pluspluscolor='blue')
    # plt.show()

    # Uncomment below to plot the evolution of an unencoded qubit.

    # one_qubit_dephasing = OneQubitDephasing(total_time=2e7, num_intervals=5)
    # one_qubit_dephasing.simulate(num_runs=100)
    # one_qubit_dephasing.compare_results_to_theory_plot()
    # plt.show()

    unittest.main()
