from netsquid_simulationtools.parameter_set import Parameter, ParameterSet
import numpy as np


def _coherence_time_prob_fn(x):
    """Computes approximate probability of no-error after one time-unit with coherence time `x`, assuming that the
    dephasing probability follows a Gaussian distribution.

    The probability that a Z error does not occur over a unit of time in a qubit with coherence time `x` is given by
    .. math::
        \\frac{1}{2} \\left(1 + e^{-2/x^2}\\right)

    To first order, this can be approximated as
    .. math::
        e^{-1/x^2}

    Parameters
    ----------
    x : float
        Coherence time.

    Returns
    -------
    float
        Approximate probability of no-error.

    """

    if x == 0.:
        return 0.
    else:
        return np.exp(-1. / x ** 2)


def _inverse_coherence_time_prob_fn(x):
    """Inverse of `_coherence_time_prob_fn`. Computes coherence time from corresponding probability of no-error.

    Parameters
    ----------
    x : float
        Probability of no-error.

    Returns
    -------
    float
        Coherence time.

    """
    if x == 1.:
        return np.inf
    elif x == 0.:
        return 0.
    else:
        return np.sqrt(- 1. / np.log(x))


class TIParameterSet(ParameterSet):

    _REQUIRED_PARAMETERS = [

        Parameter(name="visibility",
                  units=None,
                  perfect_value=1.,
                  type=float,
                  convert_to_prob_fn=lambda x: x,
                  convert_from_prob_fn=lambda x: x),

        Parameter(name="coin_prob_ph_ph",
                  units=None,
                  perfect_value=1.,
                  type=float,
                  convert_to_prob_fn=lambda x: x,
                  convert_from_prob_fn=lambda x: x),

        Parameter(name="dark_count_probability",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x),

        Parameter(name="collection_efficiency",
                  units=None,
                  perfect_value=1.,
                  type=float,
                  convert_to_prob_fn=lambda x: x,
                  convert_from_prob_fn=lambda x: x),

        Parameter(name="detector_efficiency",
                  units=None,
                  perfect_value=1.,
                  type=float,
                  convert_to_prob_fn=lambda x: x,
                  convert_from_prob_fn=lambda x: x),

        Parameter(name="coherence_time",
                  units="ns",
                  perfect_value=np.inf,
                  type=float,
                  convert_to_prob_fn=_coherence_time_prob_fn,
                  convert_from_prob_fn=_inverse_coherence_time_prob_fn),

        Parameter(name="prob_error_0",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x),

        Parameter(name="prob_error_1",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x),

        Parameter(name="ms_depolar_prob",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x),

        Parameter(name="rot_z_depolar_prob",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x),

        Parameter(name="emission_fidelity",
                  units=None,
                  perfect_value=1.,
                  type=float,
                  convert_to_prob_fn=lambda x: x,
                  convert_from_prob_fn=lambda x: x),

        Parameter(name="multi_qubit_xy_depolar_prob",
                  units=None,
                  perfect_value=0.,
                  type=float,
                  convert_to_prob_fn=lambda x: 1 - x,
                  convert_from_prob_fn=lambda x: 1 - x)]
