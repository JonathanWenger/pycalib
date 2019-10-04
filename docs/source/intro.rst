Introduction
============

``pycalib`` is a Python package providing and comparing different methods of probability calibration
for classification.

Motivation
----------

Modeling uncertainty in a classification task is critical for many applications. However, state-of-the-art classifiers
are often not calibrated, meaning their predicted posterior probabilities do not match their empirical accuracy.
This package provides different calibration methods for multi-class problems and arbitrary classifiers. In particular it
implements a Gaussian process based calibration method for multi-class classification introduced in

.. code-block::

	@Article{wenger2019nonparametric,
	  author        = {Jonathan Wenger and Hedvig Kjellström and Rudolph Triebel},
	  title         = {Non-Parametric Calibration for Classification},
	  journal       = {arXiv preprint arXiv:1906.04933},
	  year          = {2019},
	  archiveprefix = {arXiv},
	  eprint        = {1906.04933},
	  keywords      = {calibration, non-parametric, gaussian processes, classification},
	  url           = {https://github.com/JonathanWenger/pycalib}
	}

Installation
------------
You can install this Python 3 package using `pip` (or `pip3`):

.. code-block:: console

    pip install setuptools numpy scipy scikit-learn cython
    pip install git+https://github.com/JonathanWenger/pycalib.git

Note that some dependencies need to be installed separately since our package depends on
``scikit-garden`` (https://github.com/scikit-garden/scikit-garden). Alternatively you can clone this repository with

.. code-block:: console

    pip install setuptools numpy scipy scikit-learn cython
    git clone https://github.com/JonathanWenger/pycalib
    cd pycalib
    python setup.py install

Requirements
------------
``pycalib`` was developed under Python 3.6. Any required Python packages should be automatically installed when using
``pip``. If not using pip, check the ``requirements.txt`` at https://github.com/JonathanWenger/pycalib.