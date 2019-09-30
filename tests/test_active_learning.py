import pytest
import numpy as np
import pycalib.active_learning as al


def test_query_normalization():
    # Check for query criterion in [0, 1]
    p_pred = np.ones([100, 10]) / 10
    assert (np.all(al.query_norm_entropy(p_pred) <= 1) and np.all(
        al.query_norm_entropy(p_pred) >= 0)), "Query criterion values outside of [0,1]"
