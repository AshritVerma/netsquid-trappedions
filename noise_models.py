import random
from scipy.linalg import expm
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.operators import Operator, Z, I


class CollectiveDephasingNoiseModel(QuantumErrorModel):
    """
    Model for applying collective dephasing noise to qubit(s) on an ion trap.
    """

    def __init__(self, coherence_time):
        super().__init__()
        self.required_properties.append('dephasing_rate')
        self.coherence_time = coherence_time

    def error_operation(self, qubits, delta_time=0, **properties):
        """Noisy quantum operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubit has spent on component [ns].

        """
        dephasing_rate = properties['dephasing_rate']
        coherence_time = self.coherence_time
        if coherence_time != 0.:
            rotation_matrix = expm(1j * dephasing_rate * delta_time / coherence_time * Z.arr)
        else:
            rotation_matrix = I
        rotation_operator = Operator(name='rotation_operator', matrix=rotation_matrix,
                                     description='collective dephasing operator, rotation rate has been sampled')
        for qubit in qubits:
            qapi.operate(qubit, rotation_operator)


class EmissionNoiseModel(QuantumErrorModel):
    """Used to model noise and loss of the emission of an (entangled) photon.

    Parameters
    ----------
    emission_fidelity : float in the closed interval [0.25, 1]
        If the emitted qubit is in the Phi+ Bell state with some other qubit,
        the noise model will turn it into a Werner state with fidelity emission_fidelity to the Phi+ Bell state.
        Here, fidelity is understood according to the "squared" definition.
    collection_efficiency: float in the closed interval [0, 1]
        Probability that the qubit is not lost during emission.

    Notes
    -----
        Noise model must be assigned as qout_noise_model to a quantum memory, since it uses the pop() method.

        Designed primarily for use with INSTR_EMIT. Can also be used for other emission processes,
        but perhaps emission_fidelity is then not defined as conveniently.

        qapi.depolarize is used for noise. The depolarization probability is related to emission_fidelity by
        P = 4 / 3 * (1 - F).
    """

    def __init__(self, emission_fidelity, collection_efficiency, **kwargs):
        super().__init__(**kwargs)
        self.fidelity = emission_fidelity
        self.collection_efficiency = collection_efficiency

    def error_operation(self, qubits, delta_time=0, **properties):
        """Noisy quantum operation to apply to qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to. This is the qubit that it being emitted.
        delta_time : float, optional
            Always zero because this noise model is used as qin/qout noise model.
        """

        if random.random() > self.collection_efficiency:

            qapi.discard(qubits[0])

        if qubits[0] is not None and qubits[0].qstate is not None:
            # only apply noise if a photon is emitted

            # depolarizing a Bell state with this probability gives Werner state with specified fidelity
            depol_prob = 4 / 3 * (1 - self.fidelity)

            # note that depolarizing one qubit in Bell state is equivalent to depolarizing the two-qubit state
            qapi.depolarize(qubits[0], depol_prob)
