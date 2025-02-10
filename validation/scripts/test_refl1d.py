import itertools
import warnings
import pytest
import numpy as np
from test_discovery import get_test_data, get_polarised_test_data

from refl1d.probe import abeles
from refl1d.sample.reflectivity import reflectivity_amplitude, magnetic_amplitude
from refl1d.names import Stack, QProbe, Experiment, SLD

# abeles.refl is a Python calculator, reflectivity_amplitude uses
# a C extension
backends = [abeles.refl, reflectivity_amplitude]
tests_backends = list(itertools.product(get_test_data(), backends))
ids = [f"{t[0][0]}-{t[1]}" for t in tests_backends]


@pytest.mark.parametrize("nsd, backend", tests_backends, ids=ids)
def test_refl1d(nsd, backend):
    """
    Run validation for refl1d.

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    test_name, slabs, data = nsd

    if data.shape[1] == 4:
        # resolution smeared
        if backend == reflectivity_amplitude:
            # no way of setting backend for resolution smearing tests
            return

        resolution_test(slabs, data)
    elif data.shape[1] < 4:
        # no resolution data, just test kernel
        kernel_test(slabs, data, backend)


def kernel_test(slabs, data, backend):
    """
    Test the reflectivity kernels for ref1d.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
    """
    r = backend(
        data[:, 0] / 2.0,
        slabs[:, 0],
        slabs[:, 1],
        irho=slabs[:, 2],
        sigma=slabs[1:, 3],
    )
    R = (r * np.conj(r)).real

    assert R.shape == data[:, 1].shape
    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


def resolution_test(slabs, data):
    stk = Stack()
    for i, slab in enumerate(slabs[::-1]):
        m = SLD(f"layer {i}", rho=slab[1], irho=slab[2])
        stk |= m(thickness=slab[0], interface=slab[-1])

    probe = QProbe(Q=data[:, 0], dQ=data[:, 3])
    probe.oversample(21, seed=1)

    try:
        M = Experiment(stk, probe)
        _, R = M.reflectivity()
        np.testing.assert_allclose(R, data[:, 1], rtol=0.033)
    except AssertionError:
        # Probe oversampling did not work.
        # make our own oversampling with a linearly spaced array
        warnings.warn(
            "QProbe oversampling didn't work. Trying linearly spaced points",
            RuntimeWarning,
        )
        argmin = np.argmin(data[:, 0])
        argmax = np.argmax(data[:, 0])

        probe.calc_Qo = np.linspace(
            data[argmin, 0] - 3.5 * data[argmin, 3],
            data[argmax, 0] + 3.5 * data[argmax, 3],
            21 * len(data),
        )
        M = Experiment(stk, probe)

        _, R = M.reflectivity()
        np.testing.assert_allclose(R, data[:, 1], rtol=0.033)


pol_backends = [
    magnetic_amplitude,
]
pol_tests_backends = list(
    itertools.product(get_polarised_test_data(), pol_backends)
)
pol_ids = [f"{t[0][0]}-{t[1]}" for t in pol_tests_backends]


@pytest.mark.parametrize("nsd, backend", pol_tests_backends, ids=pol_ids)
def test_pol_refl1d(nsd, backend):
    """
    Run validation for refl1d (polarised beam).

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data, AGUIDE, H
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
    """
    test_name, slabs, data, AGUIDE, H = nsd

    Rmm, Rmp, Rpm, Rpp = pol_kernel_test(slabs[::-1], data, AGUIDE, H, backend)


def pol_kernel_test(slabs, data, AGUIDE, H, backend):
    """
    Test the reflectivity kernels for ref1d.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    backend: {abeles.refl, reflectivity_amplitude}
        function for reflectance calculation
    """
    kz = data[:, 0] / 2.0  # = Qz / 2
    rmm, rmp, rpm, rpp = backend(
        kz,
        slabs[:, 0],  # thickness
        slabs[:, 1],  # SLDn
        irho=slabs[:, 2],  # SLDi,
        thetaM=slabs[:, 3],  # thetaM
        rhoM=slabs[:, 4],  # SLDm,
        sigma=slabs[:-1, 5],  # sigma
        Aguide=AGUIDE,  # AGUIDE
        H=H,  # applied field H
    )
    Rmm, Rmp, Rpm, Rpp = [(r * np.conj(r)).real for r in [rmm, rmp, rpm, rpp]]
    assert Rmm.shape == data[:, 1].shape

    np.testing.assert_allclose(Rmm, data[:, 1], rtol=8e-5, atol=1e-12)
    np.testing.assert_allclose(Rmp, data[:, 2], rtol=8e-5, atol=1e-12)
    np.testing.assert_allclose(Rpm, data[:, 3], rtol=8e-5, atol=1e-12)
    np.testing.assert_allclose(Rpp, data[:, 4], rtol=8e-5, atol=1e-12)
    return Rmm, Rmp, Rpm, Rpp


if __name__ == "__main__":
    for nsd, backend in tests_backends:
        test_refl1d(nsd, backend)
    for nsd, backend in pol_tests_backends:
        test_pol_refl1d(nsd, backend)
