import os

import pytest
import numpy as np


def _skip_R(reason):
    # with COXDEV_REQUIRE_R set (as in CI), a missing R setup fails the tests
    # that need it instead of skipping them
    if os.environ.get('COXDEV_REQUIRE_R'):
        pytest.fail(reason)
    pytest.skip(reason)

def pytest_addoption(parser):
    parser.addoption(
        "--test-size",
        type=int,
        default=-1,
        help="test size: how many of cumsum and compareR tests to run (-1 indicates all)",
    )


@pytest.fixture(scope='session')
def Rinfo():
    val = {}
    try:
        import rpy2.robjects as rpy
        val['rpy'] = rpy
        val['has_rpy2'] = True
    except ImportError:
        _skip_R('rpy2 is not importable')

    from rpy2.robjects.packages import importr, PackageNotInstalledError
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.vectors import FloatVector, IntVector
    from rpy2.robjects import Formula
    from rpy2.robjects import DataFrame

    val['rpy'] = rpy
    val['importr'] = importr
    val['numpy2ri'] = numpy2ri
    val['default_converter'] = default_converter
    val['FloatVector'] = FloatVector
    val['IntVector'] = IntVector
    val['Formula'] = Formula
    val['DataFrame'] = DataFrame

    val['np_cv_rules'] = default_converter + numpy2ri.converter + pandas2ri.converter
    np_cv_rules = val['np_cv_rules']
    
    try:
        val['glmnetR'] = importr('glmnet')
        val['survivalR'] = importr('survival')
    except PackageNotInstalledError as e:
        _skip_R(f'an R package the tests need is not installed: {e}')
    val['baseR'] = importr('base')
    val['statR'] = importr('stats')

    return val

def pytest_collection_modifyitems(config, items):
    test_size = config.getoption("--test-size")
    if test_size >= 0:
        rng = np.random.default_rng(0)

        compare_R = []
        cumsum = []
        zero_weights = []
        jax_tie_types = []
        new_items = []

        for item in items:
            # We only cap tests from tests/test_compareR.py
            if "test_compareR.py" in str(item.fspath): 
                compare_R.append(item)
            elif "test_cumsums.py" in str(item.fspath):
                cumsum.append(item)
            elif "test_zero_weights.py" in str(item.fspath):
                zero_weights.append(item)
            elif "test_jax_all_tie_types" in item.name:
                jax_tie_types.append(item)
            else:
                new_items.append(item)
        if len(compare_R) > 0:
            chosen = set(rng.choice(compare_R, test_size, replace=True))
            new_items += [item for item in compare_R if item in chosen]   # in collection order
        if len(cumsum) > 0:
            chosen = set(rng.choice(cumsum, test_size, replace=True))
            new_items += [item for item in cumsum if item in chosen]   # in collection order
        if len(zero_weights) > 0:
            chosen = set(rng.choice(zero_weights, test_size, replace=True))
            new_items += [item for item in zero_weights if item in chosen]   # in collection order
        if len(jax_tie_types) > 0:
            chosen = set(rng.choice(jax_tie_types, test_size, replace=True))
            new_items += [item for item in jax_tie_types if item in chosen]   # in collection order
        items[:] = new_items

