import numpy as np
from netsquid.components import QuantumProgram, INSTR_INIT
from netsquid.components.instructions import INSTR_ROT_Z, INSTR_MEASURE, INSTR_EMIT
from netsquid_trappedions.instructions import IonTrapMSGate, IonTrapMultiQubitRotation
from netsquid.qubits.ketstates import BellIndex

ms_instruction = IonTrapMSGate(2, np.pi / 2)


class IonTrapSwapProgram(QuantumProgram):
    """
    Internal working
    ----------------
    A few private attributes:
      * _NAME_OUTCOME_CONTROL : str
      * _NAME_OUTCOME_TARGET : str
      * _OUTCOME_TO_BELL_INDEX : dict with keys (int, int) and values :class:`netsquid.qubits.ketstates.BellIndex`

           Indicates how the two measurement outcomes are related to the
           state that is measured. Its keys are tuples of the two measurement
           outcomes (control, target) and its values is the Bell state index.
    """

    default_num_qubits = 2
    _NAME_OUTCOME_CONTROL = "control-qubit-outcome"
    _NAME_OUTCOME_TARGET = "target-qubit-outcome"
    _OUTCOME_TO_BELL_INDEX = {(1, 1): BellIndex.PHI_PLUS, (0, 1): BellIndex.PSI_PLUS,
                              (1, 0): BellIndex.PSI_MINUS, (0, 0): BellIndex.PHI_MINUS}
    keep_measured_qubits = False

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(INSTR_ROT_Z, q1, angle=np.pi / 4)
        self.apply(INSTR_ROT_Z, q2, angle=-np.pi / 4)
        self.apply(ms_instruction, qubit_indices=[q1, q2])
        self.apply(INSTR_MEASURE, q1, output_key=self._NAME_OUTCOME_CONTROL, keep=self.keep_measured_qubits)
        self.apply(INSTR_MEASURE, q2, output_key=self._NAME_OUTCOME_TARGET, keep=self.keep_measured_qubits)
        yield self.run()
        self.output["bell_index"] = self.get_outcome_as_bell_index

    @property
    def get_outcome_as_bell_index(self):
        m_outcome_control = self.output[self._NAME_OUTCOME_CONTROL][0]
        m_outcome_target = self.output[self._NAME_OUTCOME_TARGET][0]
        return self._OUTCOME_TO_BELL_INDEX[(m_outcome_control, m_outcome_target)]


class IonTrapOneQubitHadamard(QuantumProgram):

    default_num_qubits = 1

    def program(self):
        q = self.get_qubit_indices()
        self.apply(instruction=INSTR_ROT_Z, qubit_indices=q, angle=np.pi)
        self.apply(IonTrapMultiQubitRotation(num_positions=1), qubit_indices=q, phi=np.pi / 2, theta=np.pi / 2)
        yield self.run()


class EmitProg(QuantumProgram):

    default_num_qubits = 2

    def program(self):
        memory_position, emission_position = self.get_qubit_indices()
        self.apply(instruction=INSTR_INIT, qubit_indices=memory_position)
        self.apply(instruction=INSTR_EMIT, qubit_indices=[memory_position, emission_position])
        yield self.run()


def ion_trap_meas_prog(meas_basis):

    if meas_basis != "X" and meas_basis != "Z":
        raise ValueError("Measurement basis should be either X or Z")
    prog = QuantumProgram(num_qubits=1, parallel=False)
    q = prog.get_qubit_indices()
    if meas_basis == "X":
        prog.apply(instruction=INSTR_ROT_Z, qubit_indices=q, angle=np.pi)
        prog.apply(IonTrapMultiQubitRotation(num_positions=1), qubit_indices=q, phi=np.pi / 2,
                   theta=np.pi / 2)
    prog.apply(INSTR_MEASURE, qubit_indices=q, output_key="outcome")

    return prog


ion_trap_meas_z = ion_trap_meas_prog("Z")
ion_trap_meas_x = ion_trap_meas_prog("X")
emit_prog = EmitProg()
ion_trap_one_qubit_hadamard = IonTrapOneQubitHadamard()
ion_trap_swap_program = IonTrapSwapProgram(num_qubits=2)
