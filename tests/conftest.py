import pytest
import numpy as np

def pytest_addoption(parser):
    parser.addoption(
        "--test-size",
        type=int,
        default=-1,
        help="test size: how many of cumsum and compareR tests to run (-1 indicates all)",
    )

def pytest_collection_modifyitems(config, items):
    test_size = config.getoption("--test-size")
    if test_size >= 0:
        rng = np.random.default_rng(0)

        compare_R = []
        cumsum = []
        zero_weights = []
        new_items = []

        for item in items:
            # We only cap tests from tests/test_compareR.py
            if "test_compareR.py" in str(item.fspath): 
                compare_R.append(item)
            elif "test_cumsums.py" in str(item.fspath):
                cumsum.append(item)
            elif "test_zero_weights.py" in str(item.fspath):
                zero_weights.append(item)
            else:
                new_items.append(item)
        if len(compare_R) > 0:
            new_items += list(set(rng.choice(compare_R, test_size, replace=True)))
        if len(cumsum) > 0:
            new_items += list(set(rng.choice(cumsum, test_size, replace=True)))
        if len(zero_weights) > 0:
            new_items += list(set(rng.choice(zero_weights, test_size, replace=True)))
        items[:] = new_items

