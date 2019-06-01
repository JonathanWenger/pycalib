import numpy as np
import GPy
from GPy.testing.link_function_tests import LinkFunctionTests
from GPy.testing.mapping_tests import MappingGradChecker
from pycalib import gpy_classes

# Mean function
def test_logmapping():
    mapping = gpy_classes.Log(3, 3)
    X = np.abs(np.random.randn(100, 3)) + 10**-6
    assert MappingGradChecker(mapping, X).checkgrad()

# Inverse Link function
def test_softargmax_gradients():
        # transf dtransf_df d2transf_df2 d3transf_df3
        link = gpy_classes.Softargmax()
        _lim_val = np.finfo(np.float64).max
        _lim_val_exp = np.log(_lim_val)
        lim_of_inf = _lim_val_exp
        lft = LinkFunctionTests()
        lft.setUp()
        lft.check_gradient(link, lim_of_inf, test_lim=True)

# Likelihood

# Inference

# Prediction

