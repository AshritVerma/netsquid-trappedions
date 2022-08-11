from netsquid_trappedions.ti_parameter_set import TIParameterSet, _coherence_time_prob_fn, \
    _inverse_coherence_time_prob_fn
import numpy as np


def test_perfect_hardware():

    class TIParameterSetPerfect(TIParameterSet):
        visibility = 1.
        coin_prob_ph_ph = 1.
        dark_count_probability = 0.
        collection_efficiency = 1.
        detector_efficiency = 1.
        coherence_time = np.inf
        prob_error_0 = 0.
        prob_error_1 = 1.
        ms_depolar_prob = 0.
        rot_z_depolar_prob = 0.
        emission_fidelity = 1.
        multi_qubit_xy_depolar_prob = 1.

    ti_perfect_parameters = TIParameterSetPerfect()

    assert ti_perfect_parameters.visibility == 1.
    assert ti_perfect_parameters.coin_prob_ph_ph == 1.
    assert ti_perfect_parameters.dark_count_probability == 0.
    assert ti_perfect_parameters.collection_efficiency == 1.
    assert ti_perfect_parameters.detector_efficiency == 1.
    assert ti_perfect_parameters.coherence_time == np.inf
    assert ti_perfect_parameters.prob_error_0 == 0.
    assert ti_perfect_parameters.prob_error_1 == 1.
    assert ti_perfect_parameters.ms_depolar_prob == 0.
    assert ti_perfect_parameters.rot_z_depolar_prob == 0.
    assert ti_perfect_parameters.emission_fidelity == 1.
    assert ti_perfect_parameters.multi_qubit_xy_depolar_prob == 1.


def test_coherence_time_prob_fn():
    # infinite coherence time means no errors
    assert _coherence_time_prob_fn(np.inf) == 1.
    assert _inverse_coherence_time_prob_fn(1.) == np.inf

    # no coherence means always an error
    assert _coherence_time_prob_fn(0.) == 0.
    assert np.isclose(_coherence_time_prob_fn(1E-10), 0.)
    assert _inverse_coherence_time_prob_fn(0.) == 0.

    # when time / coherence time = 1, should be 1 / e probability of no error
    assert _coherence_time_prob_fn(1.) == np.exp(-1.)
    assert _inverse_coherence_time_prob_fn(np.exp(-1.)) == 1.
