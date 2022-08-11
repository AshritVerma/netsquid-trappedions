import unittest
from copy import deepcopy
from random import random

import numpy as np

import netsquid as ns
import netsquid.qubits.qubitapi as qapi
from netsquid import BellIndex
from netsquid.components.instructions import INSTR_INIT, INSTR_ROT_Z, INSTR_EMIT, INSTR_MEASURE
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import create_qubits, ketstates, operate, fidelity
from netsquid.qubits.qubitapi import ops
from netsquid.components.qprocessor import MissingInstructionError
from netsquid_trappedions.instructions import IonTrapMSGate, INSTR_INIT_RANDOM, INSTR_INIT_BELL
from netsquid_trappedions.instructions import IonTrapMultiQubitRotation
from netsquid_trappedions.ion_trap import IonTrap


class TestIonTrap(unittest.TestCase):
    """
    Base testing class
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.ntry = 6
        cls.qubit_test_numbers = [1, 2, 5]
        cls.qref = []
        cls.qresult = []
        ns.qubits.qformalism.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    def setUp(self) -> None:
        ns.sim_reset()

    def tearDown(self) -> None:
        ns.sim_stop()

    def check_equal(self):
        self.assertTrue(np.all(np.isclose(self.qref, self.qresult, atol=.0000001)))
        # self.assertTrue(np.array_equal(self.qref, self.qresult))

    def check_close(self):
        self.assertFalse(np.all(np.isclose(self.qref, self.qresult, atol=.0000001)))
        self.assertTrue(np.all(np.isclose(self.qref, self.qresult, atol=.1)))
        # self.assertFalse(np.array_equal(self.qref, self.qresult))


class TestIonTrapMultiQubitGateSMatrix(TestIonTrap):

    def test_construct_s_1(self):
        instr = IonTrapMultiQubitRotation(num_positions=1)
        instr.construct_s(phi=np.pi / 4)
        expected_mtx = 1 / np.sqrt(2) * np.array([[0, 1 - 1j], [1 + 1j, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))
        instr.construct_s(phi=0)
        expected_mtx = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))
        instr.construct_s(phi=np.pi / 2)
        expected_mtx = np.array([[0, -1j], [1j, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))

    def test_construct_s_2(self):
        instr = IonTrapMultiQubitRotation(num_positions=2)
        instr.construct_s(phi=np.pi / 4)
        expected_mtx = 1 / np.sqrt(2) * np.array(
            [[0, 1 - 1j, 1 - 1j, 0], [1 + 1j, 0, 0, 1 - 1j], [1 + 1j, 0, 0, 1 - 1j], [0, 1 + 1j, 1 + 1j, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))
        instr.construct_s(phi=0)
        expected_mtx = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))
        instr.construct_s(phi=np.pi / 2)
        expected_mtx = np.array([[0, -1j, -1j, 0], [1j, 0, 0, -1j], [1j, 0, 0, -1j], [0, 1j, 1j, 0]])
        self.assertTrue(np.allclose(instr._smatrix, expected_mtx))


class TestIonTrapMultiQubitRotation(TestIonTrap):

    def test_full_rotation(self):
        """
        Check if we get back to the start for a theta=2pi rotation for any random phi.
        Also check if we split the rotation into multiple parts.
        """
        for num_qubits in self.qubit_test_numbers:
            qubit_indices = list(range(num_qubits))
            ion_trap = IonTrap(num_positions=num_qubits)
            ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
            rotation = IonTrapMultiQubitRotation(num_qubits)
            for steps in [1, 3, 5]:
                for _ in range(self.ntry):
                    phi = random() * np.pi * 2
                    prog = QuantumProgram(num_qubits=num_qubits)
                    for _ in range(steps):
                        prog.apply(instruction=rotation, qubit_indices=qubit_indices, theta=np.pi * 2 / steps, phi=phi)
                    ion_trap.execute_instruction(instruction=INSTR_INIT_RANDOM,
                                                 qubit_mapping=qubit_indices, standard_states=True)
                    ns.sim_run()
                    self.qref = deepcopy(qapi.reduced_dm(ion_trap.peek(qubit_indices)))
                    ion_trap.execute_program(prog)
                    ns.sim_run()
                    self.qresult = qapi.reduced_dm(ion_trap.peek(qubit_indices))
                    self.check_equal()

    def test_full_rotation_noise(self):
        """Check that the rotation is noisy."""
        for num_qubits in self.qubit_test_numbers:
            qubit_indices = list(range(num_qubits))
            ion_trap = IonTrap(num_positions=num_qubits, multi_qubit_xy_rotation_depolar_prob=0.001)
            ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
            rotation = IonTrapMultiQubitRotation(num_qubits)
            for _ in range(self.ntry):
                phi = random() * np.pi * 2
                ion_trap.execute_instruction(instruction=INSTR_INIT_RANDOM,
                                             qubit_mapping=qubit_indices, standard_states=True)
                ns.sim_run()
                self.qref = deepcopy(qapi.reduced_dm(ion_trap.peek(qubit_indices)))
                ion_trap.execute_instruction(instruction=rotation, qubit_mapping=qubit_indices,
                                             theta=np.pi * 2, phi=phi)
                ns.sim_run()
                self.qresult = qapi.reduced_dm(ion_trap.peek(qubit_indices))
                self.check_close()


class TestIonTrapMSGate(TestIonTrap):

    def setUp(self) -> None:
        super().setUp()
        self.qubit_test_numbers = [2, 4, 6, 8]

    def one_qubit(self, noiseless=True):
        """Check if we apply our gate it doesn't change that much. I think."""
        ms_depolar_prob = 0 if noiseless else 0.01
        ion_trap = IonTrap(num_positions=1, ms_optimization_angle=np.pi * 8, ms_depolar_prob=ms_depolar_prob)
        ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
        ms_gate = IonTrapMSGate(num_positions=1, theta=np.pi * 8)
        for _ in range(self.ntry):
            ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=True)
            ns.sim_run()
            self.qref = deepcopy(qapi.reduced_dm(ion_trap.peek([0])))
            ion_trap.execute_instruction(ms_gate, [0], phi=np.pi / 4)
            ns.sim_run()
            self.qresult = qapi.reduced_dm(ion_trap.peek([0]))
            self.check_equal() if noiseless else self.check_close()

    def test_one_qubit_noiseless(self):
        self.one_qubit(noiseless=True)

    def test_one_qubit_noisy(self):
        self.one_qubit(noiseless=False)

    def test_fully_entangling(self):
        for num_qubits in self.qubit_test_numbers:
            # Create fully entangled state manually
            qubits = qapi.create_qubits(num_qubits)
            qapi.operate([qubits[0]], ops.H)
            for qubit in qubits[1:]:
                qapi.operate([qubits[0], qubit], ops.CNOT)
            self.qref = qapi.reduced_dm(qubits)

            # Create fully entangled state using a program
            qubit_indices = list(range(num_qubits))
            ion_trap = IonTrap(num_positions=num_qubits, ms_optimization_angle=np.pi / 2)
            ms_gate = IonTrapMSGate(num_positions=num_qubits, theta=np.pi / 2)

            prog = QuantumProgram(num_qubits=num_qubits)
            for qubit in qubit_indices:
                prog.apply(instruction=INSTR_INIT, qubit_indices=[qubit])
            # We choose a smart phi so we always end up in the GHZ state.
            # TODO: Figure out why this doesn't coincide with literature.
            prog.apply(ms_gate, qubit_indices=qubit_indices, phi=np.pi / 2 - np.pi / (2 * num_qubits))
            ion_trap.execute_program(prog)
            ns.sim_run()
            self.qresult = qapi.reduced_dm(ion_trap.peek(qubit_indices))
            self.check_equal()


class TestIInitRandom(unittest.TestCase):
    """
    Test behaviour of the random instruction
    """

    def setUp(self) -> None:
        ns.sim_reset()

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_create_0_qubits(self):
        """It should fail when we have no positions available"""
        ion_trap = IonTrap(num_positions=-1)  # with 0, there is still an emission position
        ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
        self.assertRaises(ValueError, ion_trap.execute_instruction, INSTR_INIT_RANDOM, standard_states=True)

    def test_create_1_qubits(self):
        """We should occupy the state when we are done"""
        ion_trap = IonTrap(num_positions=1)
        ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
        ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=True)
        ns.sim_run()
        self.assertTrue(ion_trap.get_position_used())

        ion_trap.reset()
        # ion_trap = create_ion_trap(num_positions=1)
        # ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
        ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=False)
        ns.sim_run()
        self.assertTrue(ion_trap.get_position_used())

    def test_create_n_qubits(self):
        """We should occupy all the states in the memory"""
        for n in [2, 3, 4, 10]:
            ion_trap = IonTrap(num_positions=n)
            ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
            ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=True)
            ns.sim_run()
            self.assertTrue(ion_trap.get_position_used())
            # ion_trap = create_ion_trap(num_positions=n)
            ion_trap.reset()
            ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=False)
            ns.sim_run()
            for i in range(n):
                self.assertTrue(ion_trap.get_position_used(i))

    def test_fair_distribution(self):
        """The distribution of the standard states should be uniform on average"""
        states = []
        ion_trap = IonTrap(num_positions=1)
        ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
        for i in range(1000):
            ion_trap.execute_instruction(INSTR_INIT_RANDOM, standard_states=True)
            ns.sim_run()
            qubit = ion_trap.peek(0)
            states.append(qubit[0].qstate.dm)
        dm = np.mean(states, axis=0)
        dm_error = np.std(states, axis=0) / np.sqrt(len(states))
        # check if we get maximally mixed state up to 3 standard deviations of the mean
        self.assertAlmostEqual(dm[0][0], .5, delta=3 * dm_error[0][0])
        self.assertAlmostEqual(dm[0][1], 0, delta=3 * dm_error[0][0])


class TestIInitBell(unittest.TestCase):

    def setUp(self) -> None:
        ns.sim_reset()
        self.ion_trap = IonTrap(num_positions=2)
        self.ion_trap.add_instruction(instruction=INSTR_INIT_BELL, duration=0)
        self.q1, self.q2 = qapi.create_qubits(2)
        ns.qubits.operate(self.q1, ops.H)
        ns.qubits.operate([self.q1, self.q2], ops.CNOT)

    def tearDown(self) -> None:
        ns.sim_run()
        dm_ref = qapi.reduced_dm([self.q1, self.q2])
        dm_ion_trap = qapi.reduced_dm(self.ion_trap.peek([0, 1]))
        np.testing.assert_array_equal(dm_ref, dm_ion_trap)
        ns.sim_stop()

    def test_phiplus(self):
        self.ion_trap.execute_instruction(INSTR_INIT_BELL, bell_index=BellIndex.PHI_PLUS)

    def test_psiplus(self):
        self.ion_trap.execute_instruction(INSTR_INIT_BELL, bell_index=BellIndex.PSI_PLUS)
        ns.qubits.operate(self.q2, ops.X)

    def test_phimin(self):
        self.ion_trap.execute_instruction(INSTR_INIT_BELL, bell_index=BellIndex.PHI_MINUS)
        ns.qubits.operate(self.q2, ops.Z)

    def test_psimin(self):
        self.ion_trap.execute_instruction(INSTR_INIT_BELL, bell_index=BellIndex.PSI_MINUS)
        ns.qubits.operate(self.q2, ops.Z)
        ns.qubits.operate(self.q2, ops.X)


class TestIonTrapZRotation(TestIonTrap):
    """
    This test is just a sanity check, but doesn't actually check anything.
    """

    def test_full_rotation(self):
        """
        Check if we get back to the start for a theta=2pi rotation for any random phi.
        Also check if we split the rotation into multiple parts.
        """
        for num_qubits in self.qubit_test_numbers:
            qubit_indices = list(range(num_qubits))
            ion_trap = IonTrap(num_positions=num_qubits)
            ion_trap.add_instruction(INSTR_INIT_RANDOM, duration=0)
            for steps in [1, 3, 6]:
                for _ in range(self.ntry):
                    prog = QuantumProgram(num_qubits=num_qubits)
                    for qubit in range(num_qubits):
                        for _ in range(0, steps):
                            prog.apply(instruction=INSTR_ROT_Z, qubit_indices=[qubit], angle=np.pi * 2 / steps)
                    ion_trap.execute_instruction(instruction=INSTR_INIT_RANDOM,
                                                 qubit_mapping=qubit_indices, standard_states=True)
                    ns.sim_run()
                    self.qref = deepcopy(qapi.reduced_dm(ion_trap.peek(qubit_indices)))
                    ion_trap.execute_program(prog)
                    ns.sim_run()
                    self.qresult = qapi.reduced_dm(ion_trap.peek(qubit_indices))
                    self.check_equal()

    def test_noise(self):
        ion_trap = IonTrap(num_positions=1, rot_z_depolar_prob=0.01)
        ion_trap.execute_instruction(INSTR_INIT, [0])
        ns.sim_run()
        self.qref = deepcopy(qapi.reduced_dm(ion_trap.peek([0])))
        ion_trap.execute_instruction(INSTR_ROT_Z, [0], angle=np.pi * 2)
        ns.sim_run()
        self.qresult = qapi.reduced_dm(ion_trap.peek([0]))
        self.check_close()

    def test_half_rotation(self):
        ion_trap = IonTrap(num_positions=1)
        self.qref = [[0, 0], [0, 1]]
        qubit = qapi.create_qubits(1)
        qapi.operate(qubits=qubit, operator=ops.H)
        ion_trap.put(qubit)
        ion_trap.execute_instruction(INSTR_ROT_Z, [0], angle=np.pi)
        ns.sim_run()
        qapi.operate(qubits=qubit, operator=ops.H)
        self.qresult = qapi.reduced_dm(qubit)
        self.check_equal()


class TestIonTrapEmission(unittest.TestCase):
    """
    This test check that when ions emit photons, this happens with a success chance between zero and one,
    and that the photon and ion have fidelity>0.5 to the phi+ Bell state.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ns.qubits.qformalism.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)
        ns.sim_reset()
        cls.name = "ion_trap"
        cls.expected_meta = {"source": cls.name}
        cls.number_of_times = 10

    def emit(self, number_of_times, ion, fidelity, collection_efficiency):
        """
        Returns a list with the fidelity of each successful emission. The number of successes is the length of the list.
        """

        # setup ion trap
        ion_trap = IonTrap(num_positions=1, collection_efficiency=collection_efficiency, emission_fidelity=fidelity)
        ion_trap.name = self.name

        fidelities = []

        for _ in range(number_of_times):

            # initialize memory in specified state
            [ion_instantiation] = qapi.create_qubits(1)
            ion_state = ion.qstate.dm.copy()
            qapi.assign_qstate([ion_instantiation], ion_state)
            ion_trap.put(ion_instantiation, positions=0)

            # execute emission
            ion_trap.execute_instruction(INSTR_EMIT, [0, ion_trap.emission_position])
            ns.sim_run()

            # receive and check message
            message = ion_trap.ports["qout"].rx_output()
            self.assertTrue(self.expected_meta.items() <= message.meta.items())
            self.assertEqual(len(message.items), 1)

            # check if a qubit was emitted and whether the state has the specified fidelity
            photon = message.items[0]
            if photon.qstate is not None:
                fidelities.append(qapi.fidelity([ion_instantiation, photon], ketstates.b00, squared=True))

            ion_trap.reset()
            ns.sim_reset()

        return fidelities

    def test_fidelity(self):

        for emission_fidelity in np.linspace(start=0.25, stop=1, num=10):
            [ion] = qapi.create_qubits(1)
            observed_fidelity = self.emit(number_of_times=1, ion=ion, fidelity=emission_fidelity,
                                          collection_efficiency=1)
            self.assertEqual(len(observed_fidelity), 1)
            self.assertAlmostEqual(emission_fidelity, observed_fidelity[0])

    def test_perfect_emission(self):

        [ion] = qapi.create_qubits(1)
        num_suc = len(self.emit(number_of_times=self.number_of_times, ion=ion, fidelity=1, collection_efficiency=1))
        self.assertEqual(num_suc, self.number_of_times)

    def test_zero_collection_efficiency(self):

        [ion] = qapi.create_qubits(1)
        num_suc = len(self.emit(number_of_times=self.number_of_times, ion=ion, fidelity=1, collection_efficiency=0))
        self.assertEqual(num_suc, 0)

    def test_half_collection_efficiency(self):

        [ion] = qapi.create_qubits(1)
        num_suc = len(self.emit(number_of_times=self.number_of_times, ion=ion, fidelity=1, collection_efficiency=.5))
        self.assertNotEqual(num_suc, 0)
        self.assertNotEqual(num_suc, self.number_of_times)

    def test_ion_unavailable(self):

        [ion] = qapi.create_qubits(1)
        qapi.operate(ion, ops.X)
        num_suc = len(self.emit(number_of_times=self.number_of_times, ion=ion, fidelity=1, collection_efficiency=1))
        self.assertEqual(num_suc, 0)

    def test_half_ion_unavailable(self):

        [ion] = qapi.create_qubits(1)
        qapi.operate(ion, ops.H)
        num_suc = len(self.emit(number_of_times=self.number_of_times, ion=ion, fidelity=1, collection_efficiency=1))
        self.assertNotEqual(num_suc, 0)
        self.assertNotEqual(num_suc, self.number_of_times)


class TestIonTrapMSGatePhi(unittest.TestCase):
    """
    This test makes sure the rotation of the MS gate is correct for two qubits.
    """

    def setUp(self) -> None:
        ns.sim_reset()
        self.MS_GATE = IonTrapMSGate(2)
        self.MS_GATE.construct_operator(phi=np.pi / 2)

    def tearDown(self) -> None:
        ns.sim_stop()

    def test_S00(self):
        # Test 1: S00 -> S00 + i S11
        qubits = create_qubits(2)
        state = (ketstates.s00 + 1j * ketstates.s11) / np.sqrt(2)
        operate(qubits, self.MS_GATE._operator)
        self.assertAlmostEqual(fidelity(qubits, state), 1)

    def test_S01(self):
        # Test 2: S01 -> S01 - i S10
        qubits = create_qubits(2)
        operate(qubits[1], ops.X)
        state = (ketstates.s01 - 1j * ketstates.s10) / np.sqrt(2)
        operate(qubits, self.MS_GATE._operator)
        self.assertAlmostEqual(fidelity(qubits, state), 1)

    def test_S10(self):
        # Test 3: S10 -> S10 - i S01
        qubits = create_qubits(2)
        operate(qubits[0], ops.X)
        state = (ketstates.s10 - 1j * ketstates.s01) / np.sqrt(2)
        operate(qubits, self.MS_GATE._operator)
        self.assertAlmostEqual(fidelity(qubits, state), 1)

    def test_S11(self):
        # Test 4: S11 -> S11 + i S00
        qubits = create_qubits(2)
        operate(qubits[0], ops.X)
        operate(qubits[1], ops.X)
        state = (ketstates.s11 + 1j * ketstates.s00) / np.sqrt(2)
        operate(qubits, self.MS_GATE._operator)
        self.assertAlmostEqual(fidelity(qubits, state), 1)


class TestEmissionPositionExclusion(unittest.TestCase):
    """Test whether instructions involving ion trap's auxiliary emission position are forbidden."""

    @classmethod
    def setUpClass(cls) -> None:
        ns.sim_reset()
        cls.small_ion_trap = IonTrap(num_positions=1)
        cls.small_ms_gate = IonTrapMSGate(num_positions=cls.small_ion_trap.num_ions,
                                          theta=cls.small_ion_trap.properties["ms_optimization_angle"])
        cls.large_ion_trap = IonTrap(num_positions=5)
        cls.large_ms_gate = IonTrapMSGate(num_positions=cls.large_ion_trap.num_ions,
                                          theta=cls.large_ion_trap.properties["ms_optimization_angle"])

    def setUp(self) -> None:
        self.small_ion_trap.execute_instruction(instruction=INSTR_INIT, qubit_mapping=[0])
        self.large_ion_trap.execute_instruction(instruction=INSTR_INIT, qubit_mapping=[0, 1, 2, 3, 4])
        ns.sim_run()

    def try_instr(self, instruction, num_positions):
        failed = False
        try:
            if num_positions == 1:
                self.small_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=[self.small_ion_trap.emission_position])
                self.large_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=[self.large_ion_trap.emission_position])
            elif num_positions == 2:
                self.small_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=[0, self.small_ion_trap.emission_position])
                self.large_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=[0, self.large_ion_trap.emission_position])
                ns.sim_run()
                self.large_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=[self.large_ion_trap.emission_position - 1,
                                                                       self.large_ion_trap.emission_position])
            elif num_positions == "all":
                self.small_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=list(range(self.small_ion_trap.num_positions)))
                self.large_ion_trap.execute_instruction(instruction=instruction,
                                                        qubit_mapping=list(range(self.large_ion_trap.num_positions)))
            else:
                raise ValueError("try_instr only works for one and two qubit gates.")
            ns.sim_run()
        except MissingInstructionError:
            failed = True
        self.assertTrue(failed)

    def test_init(self):
        self.try_instr(instruction=INSTR_INIT, num_positions=1)
        self.try_instr(instruction=INSTR_INIT, num_positions=2)
        self.try_instr(instruction=INSTR_INIT, num_positions="all")

    def test_meas(self):
        self.try_instr(instruction=INSTR_MEASURE, num_positions=1)

    def test_z_rot(self):
        self.try_instr(instruction=INSTR_ROT_Z, num_positions=1)

    def test_ms_gate(self):
        failed = False
        try:
            self.small_ion_trap.execute_instruction(instruction=self.small_ms_gate,
                                                    qubit_mapping=list(range(0, self.small_ion_trap.num_positions)))
            self.large_ion_trap.execute_instruction(instruction=self.large_ms_gate,
                                                    qubit_mapping=list(range(0, self.large_ion_trap.num_positions)))
            ns.sim_run()
            self.small_ion_trap.execute_instruction(instruction=self.small_ms_gate,
                                                    qubit_mapping=list(range(1, self.small_ion_trap.num_positions)))
            self.large_ion_trap.execute_instruction(instruction=self.large_ms_gate,
                                                    qubit_mapping=list(range(1, self.large_ion_trap.num_positions)))
            ns.sim_run()
        except MissingInstructionError:
            failed = True
        self.assertTrue(failed)

    def test_multi_qubit_xy_rotation(self):
        failed = False
        try:
            self.small_ion_trap.execute_instruction(instruction=IonTrapMultiQubitRotation(self.small_ion_trap.num_ions),
                                                    qubit_mapping=list(range(0, self.small_ion_trap.num_positions)))
            self.large_ion_trap.execute_instruction(instruction=IonTrapMultiQubitRotation(self.large_ion_trap.num_ions),
                                                    qubit_mapping=list(range(0, self.large_ion_trap.num_positions)))
            ns.sim_run()
            self.small_ion_trap.execute_instruction(instruction=IonTrapMultiQubitRotation(self.small_ion_trap.num_ions),
                                                    qubit_mapping=list(range(1, self.small_ion_trap.num_positions)))
            self.large_ion_trap.execute_instruction(instruction=IonTrapMultiQubitRotation(self.large_ion_trap.num_ions),
                                                    qubit_mapping=list(range(1, self.large_ion_trap.num_positions)))
            ns.sim_run()
        except MissingInstructionError:
            failed = True
        self.assertTrue(failed)

    def test_emission(self):
        # Note: this test is the exception, as here we test whether execution fails when the emission instruction
        # is excluded rather than included.
        failed = False
        try:
            self.large_ion_trap.execute_instruction(instruction=INSTR_EMIT,
                                                    qubit_mapping=[0, self.large_ion_trap.emission_position - 1])
            ns.sim_run()
        except MissingInstructionError:
            failed = True
        self.assertTrue(failed)


class InitializeAndMeasureProgram(QuantumProgram):

    def program(self, init_state):
        q = self.get_qubit_indices(1)
        self.apply(instruction=INSTR_INIT, qubit_indices=q)
        if init_state:
            self.apply(instruction=INSTR_ROT_Z, qubit_indices=q, angle=np.pi)
        self.apply(instruction=INSTR_MEASURE, qubit_indices=q, output_key="outcome")

        yield self.run()


def _initialize_and_measure(ion_trap, init_state=0):
    qprogram = InitializeAndMeasureProgram()
    ion_trap.execute_program(qprogram, init_state=init_state)
    ns.sim_run()

    return qprogram.output["outcome"][0]


def test_faulty_measurement():
    prob_error_0 = 1.
    prob_error_1 = 0.

    ion_trap = IonTrap(num_positions=1, prob_error_0=prob_error_0, prob_error_1=prob_error_1)

    for init_state in [0, 1]:
        # 0 measurement is always wrong, 1 is always right, so we always expect 1
        assert _initialize_and_measure(ion_trap, init_state) == 1


if __name__ == "__main__":
    unittest.main()
