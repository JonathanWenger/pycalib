import numpy as np
import pycalib

# Generate synthetic data
seed = 0
def f(x):
    return x ** 3

n_classes = 3
X, y, info_dict = pycalib.benchmark.SyntheticBeta.sample_miscal_data(alpha=.75, beta=3, miscal_func=f, miscal_func_name="power",
                                                           size=1000, marginal_probs=np.ones(n_classes) / n_classes,
                                                           random_state=seed)


