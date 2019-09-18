Examples
=============

Synthetic data
--------------
We generate some synthetic data, which represents class probabilities for a set of samples as produced by some
classifier, which is not calibrated.

.. code-block:: python

    """
    This example shows how to use the calibration methods provided by pycalib.
    """
    import numpy as np
    import sklearn
    import pycalib.benchmark as bm
    import pycalib.calibration_methods as calm

    # Generate synthetic data
    seed = 0
    def f(x):
        return x ** 3

    n_classes = 3
    X, y, info_dict = bm.SyntheticBeta.sample_miscal_data(alpha=.75, beta=3, miscal_func=f, miscal_func_name="power",
                                                               size=10000, marginal_probs=np.ones(n_classes) / n_classes,
                                                               random_state=seed)

    # Split into calibration and test data set
    p_calib, p_test, y_calib, y_test = sklearn.model_selection.train_test_split(X, y, test_size=9000, random_state=seed)

    # Train calibration method
    gpc = calm.GPCalibration(n_classes=n_classes, random_state=seed)
    gpc.fit(p_calib, y_calib)

    # Generate calibrated probabilities
    p_pred = gpc.predict_proba(p_test)


Calibration on MNIST
--------------------
The following example shows how to use ``pycalib``. We train a random forest on the benchmark dataset MNIST and
calibrate its posterior uncertainty using a multi-class calibration method.

.. code-block:: python

    """
    This example demonstrates how to calibrate a random forest trained on the MNIST benchmark dataset using pycalib.
    """
    # Package imports
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    import pycalib.calibration_methods as calm

    # Seed and data size
    seed = 0
    n_test = 10000
    n_calib = 1000

    # Download MNIST data
    X, y = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)
    X = X / 255.
    y = np.array(y, dtype=int)

    # Split into train, calibration and test data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=n_test, random_state=seed)
    X_train, X_calib, y_train, y_calib = sklearn.model_selection.train_test_split(X_train, y_train, test_size=n_calib,
                                                                                  random_state=seed)

    # Train classifier
    rf = RandomForestClassifier(random_state=seed)
    rf.fit(X_train, y_train)
    p_uncal = rf.predict_proba(X_test)

    # Predict and calibrate output
    gpc = calm.GPCalibration(n_classes=10, random_state=seed)
    gpc.fit(rf.predict_proba(X_calib), y_calib)
    p_pred = gpc.predict_proba(p_uncal)
