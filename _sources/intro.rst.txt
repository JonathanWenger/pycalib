Introduction
============

``pycalib`` is a Python package providing and comparing different methods of probability calibration
for statistical classification.

Motivation
----------

Modeling uncertainty in a classification task is critical for many applications. However, state-of-the-art classifiers
are often not calibrated, meaning their predicted posterior probabilities do not match their empirical accuracy.
This package provides different calibration methods for multi-class problems and arbitrary classifiers.

Installation
------------
You can install this python package using the command:

.. code-block:: console

    pip install git+https://github.com/JonathanWenger/pycalib

Requirements
------------
``pycalib`` was developed under Python 3.6. Any required Python packages should be automatically installed when using
``pip``. If not using pip check the file ``requirements.txt`` at https://github.com/JonathanWenger/pycalib.