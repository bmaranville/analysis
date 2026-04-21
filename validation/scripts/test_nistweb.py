import requests
import tarfile
import io
import os
import shutil
import json
import warnings
import subprocess

import pytest
import numpy as np
from test_discovery import get_test_data

tests = list(get_test_data())
ids = [f"{t[0]}" for t in tests]

LOCAL_NODE_INSTALL_PATH = "node_tmp"


@pytest.fixture(scope="module")
def local_nodeenv():
    return install_local_nodeenv()


@pytest.mark.parametrize("nsd", tests, ids=ids)
def test_nistweb(nsd, local_nodeenv):
    """
    Run validation for NIST web calculator

    Parameters
    ----------
    nsd: tuple
        test_name, slabs, data
    """
    # NCOLS of data:
    # 2 - test kernel only
    # 3 - test kernel and chi2 calculation
    # 4 - test resolution smearing and chi2 calculation

    test_name, slabs, data = nsd

    if data.shape[1] == 4:
        # resolution smeared
        warnings.warn("not testing resolution for web calculator")
    elif data.shape[1] < 4:
        # no resolution data, just test kernel
        kernel_test(test_name, slabs, data, local_nodeenv)


def kernel_test(test_name, slabs, data, local_nodeenv):
    """
    Test the reflectivity kernels for refnx.

    Parameters
    ----------
    slabs: np.ndarray
        Slab representation of the system
    data: np.ndarray
        Q, R arrays
    """
    node_bin, refl_wrapper, magrefl_wrapper = local_nodeenv
    kz = data[:, 0] / 2.0
    json_data = {
        "depth": list(slabs[:, 0]),
        "rho": list(slabs[:, 1]),
        "irho": list(slabs[:, 2]),
        "sigma": list(slabs[1:, 3]),
        "kz": list(kz),
    }

    p = subprocess.Popen(
        [node_bin, refl_wrapper], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    raw_output, raw_errors = p.communicate(json.dumps(json_data).encode())

    output = json.loads(raw_output)
    r_raw = np.array(output["R"])
    r_real = r_raw[:, 0]
    r_imag = r_raw[:, 1]
    r = r_real + 1j * r_imag
    R = (r * np.conj(r)).real

    assert R.shape == data[:, 1].shape

    np.testing.assert_allclose(R, data[:, 1], rtol=8e-5)


def install_local_nodeenv(
    path="tmp", version="v16.15.0", clean=False, force_local_node=False
):
    """returns path to node executable and refl and magrefl node wrappers"""
    current_folder = os.path.dirname(__file__)
    test_folder = os.path.join(os.path.dirname(current_folder), "test")
    working_folder = os.path.join(test_folder, path)
    os.makedirs(working_folder, exist_ok=True)

    node_exe_path = shutil.which("node")
    print(f"system node exe: {node_exe_path}")
    if node_exe_path is None or force_local_node:
        # then there is no system node installed:
        node_path = "node-{version}-linux-x64".format(version=version)
        if clean:
            os.rmtree(path)

        node_exe_path = os.path.join(working_folder, node_path, "bin", "node")
        if not os.path.exists(node_exe_path):
            # print("installing local node")
            noderaw = requests.get(
                "https://nodejs.org/dist/{version}/node-{version}-linux-x64.tar.xz".format(
                    version=version
                )
            )
            nodebytes = io.BytesIO(noderaw.content)
            nodebytes.seek(0)

            with tarfile.open(fileobj=nodebytes) as t:
                t.extractall(working_folder)

    magrefl = requests.get(
        "https://raw.githubusercontent.com/usnistgov/reflectometry-calculators/nist-pages/js/refl/magrefl.js"
    )
    open(os.path.join(working_folder, "magrefl.js"), "wb").write(
        magrefl.content
    )

    refl = requests.get(
        "https://raw.githubusercontent.com/usnistgov/reflectometry-calculators/nist-pages/js/refl/refl.js"
    )
    open(os.path.join(working_folder, "refl.js"), "wb").write(refl.content)

    shutil.copy2(
        os.path.join(current_folder, "nistweb", "magrefl_wrapper.mjs"),
        os.path.join(working_folder, "magrefl_wrapper.mjs"),
    )
    shutil.copy2(
        os.path.join(current_folder, "nistweb", "refl_wrapper.mjs"),
        os.path.join(working_folder, "refl_wrapper.mjs"),
    )
    refl_wrapper_path = os.path.join(working_folder, "refl_wrapper.mjs")
    magrefl_wrapper_path = os.path.join(working_folder, "magrefl_wrapper.mjs")
    return node_exe_path, refl_wrapper_path, magrefl_wrapper_path


if __name__ == "__main__":
    local_nodeenv = install_local_nodeenv()
    for nsd in tests:
        test_nistweb(nsd, local_nodeenv)
