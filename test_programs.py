import unittest
import logging

from netsquid import BellIndex
from netsquid.components import INSTR_INIT
from netsquid.qubits import ketstates
import netsquid.qubits.qubitapi as qapi
from netsquid_trappedions.ion_trap import IonTrap
from netsquid_trappedions.programs import IonTrapSwapProgram, ion_trap_meas_z, ion_trap_meas_x, \
    ion_trap_one_qubit_hadamard, emit_prog
from netsquid_trappedions.instructions import INSTR_INIT_BELL, IonTrapMultiQubitRotation
import netsquid as ns
import numpy as np


class TestIonTrapSwapProgram(unittest.TestCase):

    def setUp(self) -> None:
        ns.sim_reset()
        ns.qubits.qformalism.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)
        self.IonTrap = IonTrap(num_positions=2)
        self.IonTrap.add_instruction(INSTR_INIT_BELL, duration=0)
        self.SwapProgram = IonTrapSwapProgram()

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_swap(self):
        for i in range(4):
            bell_index = BellIndex(i)
            self.IonTrap.execute_instruction(INSTR_INIT_BELL, bell_index=bell_index)
            ns.sim_run()
            self.IonTrap.execute_program(self.SwapProgram)
            ns.sim_run()
            self.assertEqual(self.SwapProgram.get_outcome_as_bell_index, bell_index)


class TestIonTrapMeasurementProgram(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.IonTrap = IonTrap(num_positions=1)

    def setUp(self) -> None:
        ns.sim_reset()
        self.IonTrap.execute_instruction(INSTR_INIT, qubit_mapping=[0])
        ns.sim_run()

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_z_meas_0(self):
        self.IonTrap.execute_program(ion_trap_meas_z)
        ns.sim_run()
        result = ion_trap_meas_z.output["outcome"][0]
        self.assertEqual(result, 0)

    def test_z_meas_1(self):
        self.IonTrap.execute_instruction(IonTrapMultiQubitRotation(num_positions=1), phi=0, theta=np.pi)
        ns.sim_run()
        self.IonTrap.execute_program(ion_trap_meas_z)
        ns.sim_run()
        result = ion_trap_meas_z.output["outcome"][0]
        self.assertEqual(result, 1)

    def test_x_meas_0(self):
        self.IonTrap.execute_program(ion_trap_one_qubit_hadamard)
        ns.sim_run()
        self.IonTrap.execute_program(ion_trap_meas_x)
        ns.sim_run()
        result = ion_trap_meas_x.output["outcome"][0]
        self.assertEqual(result, 0)

    def test_x_meas_1(self):
        self.IonTrap.execute_instruction(IonTrapMultiQubitRotation(num_positions=1), phi=0, theta=np.pi)
        ns.sim_run()
        self.IonTrap.execute_program(ion_trap_one_qubit_hadamard)
        ns.sim_run()
        self.IonTrap.execute_program(ion_trap_meas_x)
        ns.sim_run()
        result = ion_trap_meas_x.output["outcome"][0]
        self.assertEqual(result, 1)


# This test should be restored

class TestIonTrapEmissionProgram(unittest.TestCase):
    """
    This test check that when ions emit photons, this happens with a success chance between zero and one,
    and that the photon and ion have fidelity>0.5 to the phi+ Bell state.
    """

    def setUp(self) -> None:
        ns.sim_reset()
        self.ion_trap = IonTrap(num_positions=1)

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_ion_trap_emission_program(self):
        self.ion_trap.execute_program(emit_prog, qubit_mapping=[0, self.ion_trap.emission_position])
        ns.sim_run()
        message = self.ion_trap.ports["qout"].rx_output()
        self.assertIsNotNone(message)
        qubit_ion = self.ion_trap.peek([0])[0]
        qubit_photon = message.items[0]
        fidelity = qapi.fidelity([qubit_ion, qubit_photon], ketstates.b00)
        self.assertAlmostEqual(fidelity, 1)


if __name__ == "__main__":
    ns.logger.setLevel(logging.DEBUG)
    unittest.main()
