# Standard imports
import pytest
import numpy as np

# Package imports
import pycalib.calibration_methods as cm


# General
@pytest.fixture(scope='module')
def sample_size():
    return 1000


@pytest.fixture(scope='module')
def p_dist_beta(sample_size, a=1, b=4):
    # Predicted probabilities (transformed to [0.5, 1])
    return np.hstack([np.random.beta(a=a, b=b, size=sample_size)]) * 0.5 + 0.5


@pytest.fixture(scope='module')
def y_cal_binary(sample_size, prob_class_0=.66):
    # Sample ground truth
    return np.random.choice(a=[0, 1], size=sample_size, replace=True, p=[prob_class_0, 1 - prob_class_0])


@pytest.fixture(scope='module')
def p_cal_binary(sample_size, y_cal_binary, p_dist_beta, a=3, b=.1, c=1):
    # Uncalibrated probabilities through miscalibration function f
    # f = lambda x: 1 / (1 + c * (1 - x) ** a / x ** b)
    f = lambda x: 1 / (1 + np.exp(-a * x - b))
    sampler_f = lambda w, y: np.random.choice(a=[1 - y, y], p=[1 - f(w), f(w)])
    y_pred = np.array(list(map(sampler_f, p_dist_beta, y_cal_binary)))

    # Compute probabilities for other classes
    p_pred = np.zeros([sample_size, 2])
    for i in range(0, 2):
        # Set probabilities for correct predictions
        correct_and_index_i = (y_pred == y_cal_binary) & (y_cal_binary == i)
        prob = p_dist_beta[correct_and_index_i]
        p_pred[correct_and_index_i, i] = prob
        p_pred[correct_and_index_i, 1 - i] = 1 - prob

        # Set probabilities for incorrect predictions
        false_and_index_i = (y_pred != y_cal_binary) & (y_cal_binary == i)
        prob = p_dist_beta[false_and_index_i]
        p_pred[false_and_index_i, i] = 1 - prob
        p_pred[false_and_index_i, 1 - i] = prob

    return p_pred


# Temperature Scaling

def test_constant_accuracy(p_cal_binary, y_cal_binary):
    # Compute accuracy
    acc = np.mean(np.equal(np.argmax(p_cal_binary, axis=1), y_cal_binary))

    # Temperature Scaling
    ts = cm.TemperatureScaling()
    ts.fit(p_cal_binary, y_cal_binary)

    # Test constant accuracy on calibration set
    p_ts = ts.predict_proba(p_cal_binary)
    acc_ts = np.mean(np.equal(np.argmax(p_ts, axis=1), y_cal_binary))
    assert acc == acc_ts, "Accuracy of calibrated probabilities does not match accuracy of calibration set."


# Histogram Binning

@pytest.mark.parametrize("binning_mode", [
    ("equal_width"),
    ("equal_freq")
])
def test_hist_binning_bin_size(p_cal_binary, y_cal_binary, binning_mode):
    n_bins = 2
    hb = cm.HistogramBinning(mode=binning_mode, n_bins=n_bins)
    hb.fit(p_cal_binary, y_cal_binary)
    assert len(hb.binning) == n_bins + 1, "Number of bins does not match input."


# Bayesian Binning into Quantiles

def test_bin_with_size_zero():
    # Data
    p_cal = .5 * np.ones([100, 2])
    y_cal = np.hstack([np.ones([50]), np.zeros([50])])

    # Fit calibration model
    bbq = cm.BayesianBinningQuantiles()
    bbq.fit(X=p_cal, y=y_cal)

    # Predict
    p_pred = bbq.predict_proba(X=p_cal)

    # Check for NaNs
    assert not np.any(np.isnan(p_pred)), "Calibrated probabilities are NaN."


# GP calibration

# OneVsAll calibration

def test_output_size_missing_classes():
    # Generate random training data with n_classes > n_calibration
    np.random.seed(1)
    n_classes = 200
    n_calibration = 100
    X_cal = np.random.uniform(0, 1, [n_calibration, n_classes])
    X_cal /= np.sum(X_cal, axis=1)[:, np.newaxis]
    y_cal = np.random.choice(range(n_classes), n_calibration)

    # Arbitrary Choice of binary calibration method
    platt = cm.PlattScaling()
    platt.fit(X_cal, y_cal)

    # Test output size
    assert np.shape(platt.predict_proba(X_cal))[
               1] == n_classes, "Predicted probabilities do not match number of classes."
