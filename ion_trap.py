import numpy as np
from netsquid.components import IMeasureFaulty, INSTR_MEASURE
from netsquid.components.qprocessor import PhysicalInstruction
from itertools import combinations
from numpy import random

from netsquid.components.models import DepolarNoiseModel
from netsquid.components.instructions import INSTR_ROT_Z, INSTR_INIT, INSTR_EMIT
from netsquid.components.qprocessor import QuantumProcessor
from netsquid_trappedions.noise_models import EmissionNoiseModel
from netsquid_trappedions.instructions import IonTrapMultiQubitRotation, IonTrapMSGate
from netsquid_trappedions.noise_models import CollectiveDephasingNoiseModel


class IonTrap(QuantumProcessor):
    """Ion-trap quantum processor capable of emitting entangled photons.

    All parameters default to their perfect values.
    That is, by default, the ion trap is initialized such that there is no noise and all gates are instantaneous.
    Note that this is unphysical.
    For realistic modelling, the ion trap should be initialized with non-perfect parameter values.

    The ionic qubits in the ion trap undergo collective dephasing in a non-Markovian manner.
    The quantum channel that describes this decoherence mechanism can be found in equation (5) of
    the paper
    "Quantum repeaters based on trapped ions with decoherence free subspace encoding",
    Zwerger et al., https://doi.org/10.1088/2058-9565/aa7983 .
    In order to model the non-Markovian process accurately, the method :meth:`IonTrap.resample` should be called
    with some regularity (e.g. every time when the qubits in the trap are reinitialized).
    For more information about this, see the documentation of :meth:`IonTrap.resample`.

    The instructions that can be performed on this QuantumProcessor are
    * Single-qubit Z rotations, :obj:`netsquid.components.instructions.INSTR_ROT_Z`.
    * Single-qubit measurements, :obj:`netsquid.components.instructions.INSTR_MEASURE`.
    * Qubit initialization, :obj:`netsquid.components.instructions.INSTR_INIT`.
    * Multi-qubit XY rotations, :class:`netsquid-trappedions.instructions.IonTrapMultiQubitRotation`.
    That is, rotations in the Bloch sphere around any axis in the XY plane can be performed,
    but only on all qubits simultaneneously.
    * Mølmer–Sørensen gate, :class:`netsquid-trappedions.instructions.IonTrapMSGate`.
    This is an entangling gate that involves all qubits in the trap.
    It can be used to prepare all the ions in a GHZ state together.
    * Emission of entangled photon, :class:`netsquid.components.instructions.INSTR_EMIT`.

    Noise on all gates is modeled as depolarizing noise.
    The ion-photon state after photon emission in the presence of noise is of the Werner form
    :math:`p |Phi+><Phi+| + (1 - p) (I / 4)`, where :math:`(I / 4)` is the maximally entangled state.

    The gate set of this ion-trap quantum processor is based on the article
    "A quantum information processor with trapped ions", Schindler et al.,
    http://stacks.iop.org/1367-2630/15/i=12/a=123012?key=crossref.cde36010c3c4d16e566cc4e802de2091 .
    The emission of photons by the ion trap is based on the article
    "Tunable ion–photon entanglement in an optical cavity", Stute et al., http://www.nature.com/articles/nature11120 .

    The number of memory positions with which this QuantumProcessor is initiated is one more than specified by
    the `num_positions` argument.
    This extra position does not represent an ionic qubit;
    instead, it is used to model the emission of an entangled photon.
    The extra position should not be accessed, and none of the gates defined above can be performed on this position.
    The number of "real" positions that represent ionic qubits can be obtained using :atr:`IonTrap.num_ions`.
    The index of the extra position can be obtained using :atr:`IonTrap.emission_position`.

    Parameters
    ----------
    num_positions : int
        Number of ions in the trap.
    coherence_time : float
        Coherence time of the qubits.
        Qubits in memory decohere according to a collective-dephasing channel characterized by this coherence time.
        This channel is both strongly correlated between all qubits in memory and non-Markovian.
    prob_error_0 : float
        Measurement error probability: probability that |0> gives outcome "1".
    prob_error_1 : float
        Measurement error probability: probability that |1> gives outcome "0".
    init_depolar_prob : float
        Parameter characterizing depolarizing channel that is applied to a qubit when it is initialized.
    rot_z_depolar_prob : float
        Parameter characterizing depolarizing channel that is applied to a qubit
        when a single-qubit z rotation is performed.
    multi_qubit_xy_rotation_depolar_prob : float
        Parameter characterizing depolarizing channel that is applied to all qubits participating in a
        multi-qubit rotation around an axis in the XY plane of the Bloch sphere.
    ms_depolar_prob : float
        Parameter charactherizing depolarizing channel that is applied to all qubits participating in a
        multi-qubit Mølmer–Sørensen gate, which is able to entangle qubits
        (this gate is the main ingredient for a Bell-state measurement).
    emission_fidelity : float
        Fidelity of the ion-photon entangled state directly after emitting a photon.
    collection_efficiency: float
        Probability that an entangled photon is successfully emitted when attempted.
    measurement_duration : float
        Time [ns] it takes to perform a single-qubit computational-basis measurement.
    initialization_duration : float
        Time [ns] it takes to initialize a qubit.
    z_rotation_duration : float
        Time [ns] it takes to perform a single-qubit z rotation.
    multi_qubit_xy_rotation_duration : float
        Time [ns] it takes to perform a multi-qubit XY rotation.
    ms_pi_over_2_duration : float
        Time [ns] it takes to perform a Mølmer–Sørensen gate with angle pi / 2.
        Durations for MS gates with larger angles are derived from this number.
    emission_duration : float
        Time [ns] it takes to attempt emitting an entangled photon.
    ms_optimization_angle : float
        Angle of Mølmer–Sørensen gate for which the device has been optimized.

    """

    def __init__(self, num_positions, coherence_time=0., prob_error_0=0., prob_error_1=0.,
                 init_depolar_prob=0., rot_z_depolar_prob=0., multi_qubit_xy_rotation_depolar_prob=0.,
                 ms_depolar_prob=0., emission_fidelity=1., collection_efficiency=1., emission_duration=0.,
                 measurement_duration=0., initialization_duration=0., z_rotation_duration=0., ms_pi_over_2_duration=0.,
                 multi_qubit_xy_rotation_duration=0., ms_optimization_angle=np.pi / 2):

        # add all parameters as properties so they can be accessed easily
        for parameter, value in locals().items():
            if parameter in ["self", "__class__", "num_positions"]:
                continue
            self.add_property(name=parameter, value=value, mutable=False)
        self.add_property(name="num_ions", value=num_positions)

        self.add_property("dephasing_rate", value=random.normal(loc=0, scale=1), mutable=True)

        super().__init__("ion_trap_quantum_communication_device",
                         # also initialize "cavity" position used for emission
                         num_positions=num_positions + 1,
                         mem_noise_models=CollectiveDephasingNoiseModel(
                             coherence_time=self.properties["coherence_time"]))

        self.resample()

        self.models["qout_noise_model"] = EmissionNoiseModel(
            emission_fidelity=self.properties["emission_fidelity"],
            collection_efficiency=self.properties["collection_efficiency"])

        # position of "cavity", memory position which should only be used to mediate emission.
        self.emission_position = num_positions
        # number of ions, i.e. "real" positions
        self.num_ions = self.num_positions - 1

        self._set_physical_instructions()

    def _set_physical_instructions(self):
        """Function that initializes the physical instructions of the ion trap.

        Physical instructions (including gates, measurements and photon emission) are initialized by this function
        using noise parameters obtained from self.properties (which are set in __init__()).

        """
        # choose topologies such that auxiliary emission position ("cavity") is excluded
        one_ion_topologies = list(range(self.num_ions))
        # two_ion_topologies = list(combinations(one_ion_topologies, 2))
        any_ion_topologies = []
        for num_qubits in range(1, self.num_ions + 1):
            for topology in combinations(one_ion_topologies, num_qubits):
                any_ion_topologies.append(topology)
        all_ion_topologies = [tuple(range(self.num_ions))]
        emit_topologies = [(ion_position, self.emission_position) for ion_position in range(self.num_ions)]

        faulty_measurement_instruction = IMeasureFaulty("faulty_z_measurement_ion_trap",
                                                        p0=self.properties["prob_error_0"],
                                                        p1=self.properties["prob_error_1"])
        physical_measurement = PhysicalInstruction(instruction=faulty_measurement_instruction,
                                                   duration=self.properties["measurement_duration"],
                                                   q_noise_model=None,
                                                   c_noise_model=None,
                                                   parallel=True,
                                                   topology=one_ion_topologies)
        self.add_composite_instruction(instruction=INSTR_MEASURE,
                                       composition=[(faulty_measurement_instruction, 0)],
                                       topology=one_ion_topologies)

        physical_init = PhysicalInstruction(instruction=INSTR_INIT,
                                            duration=self.properties["initialization_duration"],
                                            q_noise_model=DepolarNoiseModel(self.properties["init_depolar_prob"],
                                                                            time_independent=True),
                                            c_noise_model=None,
                                            parallel=True,
                                            apply_q_noise_after=True,
                                            topology=any_ion_topologies)

        z_rotation_gate = PhysicalInstruction(instruction=INSTR_ROT_Z,
                                              duration=self.properties["z_rotation_duration"],
                                              q_noise_model=DepolarNoiseModel(self.properties["rot_z_depolar_prob"],
                                                                              time_independent=True),
                                              c_noise_model=None,
                                              parallel=False,
                                              apply_q_noise_after=True,
                                              topology=one_ion_topologies)

        ms_gate = IonTrapMSGate(num_positions=self.num_ions, theta=self.properties["ms_optimization_angle"])

        ms_gate_phys = PhysicalInstruction(instruction=ms_gate,
                                           duration=5e4 * self.properties["ms_optimization_angle"] / (np.pi / 2),
                                           q_noise_model=DepolarNoiseModel(self.properties["ms_depolar_prob"],
                                                                           time_independent=True),
                                           c_noise_model=None,
                                           parallel=False,
                                           apply_q_noise_after=True,
                                           topology=all_ion_topologies)

        multi_qubit_xy_rotation = PhysicalInstruction(instruction=IonTrapMultiQubitRotation(self.num_ions),
                                                      duration=self.properties["multi_qubit_xy_rotation_duration"],
                                                      q_noise_model=DepolarNoiseModel(
                                                          self.properties["multi_qubit_xy_rotation_depolar_prob"],
                                                          time_independent=True),
                                                      c_noise_model=None,
                                                      parallel=False,
                                                      apply_q_noise_after=True,
                                                      topology=all_ion_topologies)

        emit_instruction = PhysicalInstruction(instruction=INSTR_EMIT,
                                               duration=self.properties["emission_duration"],
                                               topology=emit_topologies)

        self.add_physical_instruction(physical_measurement)
        self.add_physical_instruction(physical_init)
        self.add_physical_instruction(z_rotation_gate)
        self.add_physical_instruction(ms_gate_phys)
        self.add_physical_instruction(multi_qubit_xy_rotation)
        self.add_physical_instruction(emit_instruction)

    def resample(self):
        """Update this ion trap's properties by resampling the dephasing rate.

        By sampling the dephasing rate from a gaussian distribution, and by resampling after every experiment,
        we reproduce the statistics of equation (5) in the paper "Quantum repeaters based on trapped ions with
        decoherence free subspace encoding".

        Notes
        -----
        The dominant source of memory decoherence in ion traps is dephasing. One way of interpreting dephasing is
        as qubits undergoing Z-rotations of unknown magnitude. In ion traps, we can approximate this rotation
        to occur at a constant (but unknown) rate. There are two things that make dephasing hard to model
        in ion traps, namely

        1) the dephasing is collective, meaning that each ion dephases with (approximately) the same rate,
        2) the probability distribution over dephasing rates is Gaussian.

        The combined effect is that we have been unable to find a nice analytical solution of density matrix
        evolution when there is more than one ion. Furthermore, already for one ion, the evolution becomes
        non-Markovian, meaning that a noise model implementing the dephasing would need some sort of memory.

        All problems are solved when we don't evolve the full density matrix, but rather sample from the
        density matrix that would result from the evolution. This can be done by sampling the rotation ("dephasing")
        rate ahead of time and use this rate to constantly rotate all qubits in the trap.
        The downside is that we need to perform a simulation a large number of times before the statistics
        reproduce the density matrix in equation (5) in the paper "Quantum repeaters based on trapped ions
        with decoherence free subspace encoding" (while if we evolved the whole density matrix, the effect
        of dephasing could be extracted from the density matrix obtained after a single run).

        """

        rate = random.normal(loc=0, scale=1)
        for memory_position_num in range(self.num_positions):
            memory_position = self.subcomponents["mem_position{}".format(memory_position_num)]
            memory_position.properties.update({"dephasing_rate": rate})
        self.properties.update({"dephasing_rate": rate})

    def reset(self):
        """Reset the ion trap and resample the dephasing rate.

        """
        super().reset()
        self.resample()
