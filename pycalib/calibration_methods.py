# Standard library imports
import numpy as np
import numpy.matlib
import warnings
import matplotlib.pyplot as plt

# SciPy imports
import scipy.stats
import scipy.optimize
import scipy.special
import scipy.cluster.vq

# Scikit learn imports
import sklearn
import sklearn.multiclass
import sklearn.utils
from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed

# Calibration models
import sklearn.isotonic
import sklearn.linear_model
import betacal

# Gaussian Processes and TensorFlow
import gpflow
from pycalib import gp_classes
import tensorflow as tf

# Turn off tensorflow deprecation warnings
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

# Plotting
import pycalib.texfig

# Ignore binned_statistic FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


class CalibrationMethod(sklearn.base.BaseEstimator):
    """
    A generic class for probability calibration

    A calibration method takes a set of posterior class probabilities and transform them into calibrated posterior
    probabilities. Calibrated in this sense means that the empirical frequency of a correct class prediction matches its
    predicted posterior probability.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict(self, X):
        """
        Predict the class of new samples after scaling. Predictions are identical to the ones from the uncalibrated
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def plot(self, filename, xlim=[0, 1], **kwargs):
        """
        Plot the calibration map.

        Parameters
        ----------
        xlim : array-like
            Range of inputs of the calibration map to be plotted.

        **kwargs :
            Additional arguments passed on to :func:`matplotlib.plot`.
        """
        # TODO: Fix this plotting function

        # Generate data and transform
        x = np.linspace(0, 1, 10000)
        y = self.predict_proba(np.column_stack([1 - x, x]))[:, 1]

        # Plot and label
        plt.plot(x, y, **kwargs)
        plt.xlim(xlim)
        plt.xlabel("p(y=1|x)")
        plt.ylabel("f(p(y=1|x))")


class NoCalibration(CalibrationMethod):
    """
    A class that performs no calibration.

    This class can be used as a baseline for benchmarking.

    logits : bool, default=False
        Are the inputs for calibration logits (e.g. from a neural network)?
    """

    def __init__(self, logits=False):
        self.logits = logits

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if self.logits:
            return scipy.special.softmax(X, axis=1)
        else:
            return X


class TemperatureScaling(CalibrationMethod):
    """
    Probability calibration using temperature scaling

    Temperature scaling [1]_ is a one parameter multi-class scaling method. Output confidence scores are calibrated, meaning
    they match empirical frequencies of the associated class prediction. Temperature scaling does not change the class
    predictions of the underlying model.

    Parameters
    ----------
    T_init : float
        Initial temperature parameter used for scaling. This parameter is optimized in order to calibrate output
        probabilities.
    verbose : bool
        Print information on optimization procedure.

    References
    ----------
    .. [1] On calibration of modern neural networks, C. Guo, G. Pleiss, Y. Sun, K. Weinberger, ICML 2017
    """

    def __init__(self, T_init=1, verbose=False):
        super().__init__()
        if T_init <= 0:
            raise ValueError("Temperature not greater than 0.")
        self.T_init = T_init
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # Define objective function (NLL / cross entropy)
        def objective(T):
            # Calibrate with given T
            P = scipy.special.softmax(X / T, axis=1)

            # Compute negative log-likelihood
            P_y = P[np.array(np.arange(0, X.shape[0])), y]
            tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
            NLL = - np.sum(np.log(P_y + tiny))
            return NLL

        # Derivative of the objective with respect to the temperature T
        def gradient(T):
            # Exponential terms
            E = np.exp(X / T)

            # Gradient
            dT_i = (np.sum(E * (X - X[np.array(np.arange(0, X.shape[0])), y].reshape(-1, 1)), axis=1)) \
                   / np.sum(E, axis=1)
            grad = - dT_i.sum() / T ** 2
            return grad

        # Optimize
        self.T = scipy.optimize.fmin_bfgs(f=objective, x0=self.T_init,
                                          fprime=gradient, gtol=1e-06, disp=self.verbose)[0]

        # Check for T > 0
        if self.T <= 0:
            raise ValueError("Temperature not greater than 0.")

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        # Check is fitted
        check_is_fitted(self, "T")

        # Transform with scaled softmax
        return scipy.special.softmax(X / self.T, axis=1)

    def latent(self, z):
        """
        Evaluate the latent function Tz of temperature scaling.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence for which to evaluate the latent function.

        Returns
        -------
        f : array-like, shape=(n_evaluations,)
            Values of the latent function at z.
        """
        check_is_fitted(self, "T")
        return self.T * z

    def plot_latent(self, z, filename, **kwargs):
        """
        Plot the latent function of the calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence to plot latent function for.
        filename :
            Filename / -path where to save output.
        kwargs
            Additional arguments passed on to matplotlib.pyplot.subplots.

        Returns
        -------

        """
        check_is_fitted(self, "T")

        # Plot latent function
        fig, axes = pycalib.texfig.subplots(nrows=1, ncols=1, sharex=True, **kwargs)
        axes.plot(z, self.T * z, label="latent function")
        axes.set_ylabel("$T\\bm{z}$")
        axes.set_xlabel("$\\bm{z}_k$")
        fig.align_labels()

        # Save plot to file
        pycalib.texfig.savefig(filename)


class PlattScaling(CalibrationMethod):
    """
    Probability calibration using Platt scaling

    Platt scaling [1]_ [2]_ is a parametric method designed to output calibrated posterior probabilities for (non-probabilistic)
    binary classifiers. It was originally introduced in the context of SVMs. It works by fitting a logistic
    regression model to the model output using the negative log-likelihood as a loss function.

    Parameters
    ----------
    regularization : float, default=10^(-12)
        Regularization constant, determining degree of regularization in logistic regression.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling the data.
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    .. [1] Platt, J. C. Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood
           Methods in Advances in Large-Margin Classifiers (MIT Press, 1999)
    .. [2] Lin, H.-T., Lin, C.-J. & Weng, R. C. A note on Platt’s probabilistic outputs for support vector machines.
           Machine learning 68, 267–276 (2007)
    """

    def __init__(self, regularization=10 ** -12, random_state=None):
        super().__init__()
        self.regularization = regularization
        self.random_state = sklearn.utils.check_random_state(random_state)

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.logistic_regressor_ = sklearn.linear_model.LogisticRegression(C=1 / self.regularization,
                                                                               solver='lbfgs',
                                                                               random_state=self.random_state)
            self.logistic_regressor_.fit(X[:, 1].reshape(-1, 1), y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "logistic_regressor_")
            return self.logistic_regressor_.predict_proba(X[:, 1].reshape(-1, 1))
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class IsotonicRegression(CalibrationMethod):
    """
    Probability calibration using Isotonic Regression

    Isotonic regression [1]_ [2]_ is a non-parametric approach to mapping (non-probabilistic) classifier scores to
    probabilities. It assumes an isotonic (non-decreasing) relationship between classifier scores and probabilities.

    Parameters
    ----------
    out_of_bounds : string, optional, default: "clip"
        The ``out_of_bounds`` parameter handles how x-values outside of the
        training domain are handled.  When set to "nan", predicted y-values
        will be NaN.  When set to "clip", predicted y-values will be
        set to the value corresponding to the nearest train interval endpoint.
        When set to "raise", allow ``interp1d`` to throw ValueError.

    References
    ----------
    .. [1] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)
    .. [2] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """

    def __init__(self, out_of_bounds="clip"):
        super().__init__()
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.isotonic_regressor_ = sklearn.isotonic.IsotonicRegression(increasing=True,
                                                                           out_of_bounds=self.out_of_bounds)
            self.isotonic_regressor_.fit(X[:, 1], y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "isotonic_regressor_")
            p1 = self.isotonic_regressor_.predict(X[:, 1])
            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class BetaCalibration(CalibrationMethod):
    """
    Probability calibration using Beta calibration

    Beta calibration [1]_ [2]_ is a parametric approach to calibration, specifically designed for probabilistic
    classifiers with output range [0,1]. Here, a calibration map family is defined based on the likelihood ratio between
    two Beta distributions. This parametric assumption is appropriate if the marginal class distributions follow Beta
    distributions. The beta calibration map has three parameters, two shape parameters `a` and `b` and one location
    parameter `m`.

    Parameters
    ----------
        params : str, default="abm"
            Defines which parameters to fit and which to hold constant. One of ['abm', 'ab', 'am'].

    References
    ----------
    .. [1] Kull, M., Silva Filho, T. M., Flach, P., et al. Beyond sigmoids: How to obtain well-calibrated probabilities
           from binary classifiers with beta calibration. Electronic Journal of Statistics 11, 5052–5080 (2017)
    .. [2] Kull, M., Filho, T. S. & Flach, P. Beta calibration: a well-founded and easily implemented improvement on
           logistic calibration for binary classifiers in Proceedings of the 20th International Conference on Artificial
           Intelligence and Statistics (AISTATS)
    """

    def __init__(self, params="abm"):
        super().__init__()
        self.params = params

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.beta_calibrator_ = betacal.BetaCalibration(self.params)
            self.beta_calibrator_.fit(X[:, 1].reshape(-1, 1), y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "beta_calibrator_")
            p1 = self.beta_calibrator_.predict(X[:, 1].reshape(-1, 1))
            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class HistogramBinning(CalibrationMethod):
    """
    Probability calibration using histogram binning

    Histogram binning [1]_ is a nonparametric approach to probability calibration. Classifier scores are binned into a given
    number of bins either based on fixed width or frequency. Classifier scores are then computed based on the empirical
    frequency of class 1 in each bin.

    Parameters
    ----------
        mode : str, default='equal_width'
            Binning mode used. One of ['equal_width', 'equal_freq'].
        n_bins : int, default=20
            Number of bins to bin classifier scores into.
        input_range : list, shape (2,), default=[0, 1]
            Range of the classifier scores.

    .. [1] Zadrozny, B. & Elkan, C. Obtaining calibrated probability estimates from decision trees and naive Bayesian
           classifiers in Proceedings of the 18th International Conference on Machine Learning (ICML, 2001), 609–616.
    """

    def __init__(self, mode='equal_freq', n_bins=20, input_range=[0, 1]):
        super().__init__()
        if mode in ['equal_width', 'equal_freq']:
            self.mode = mode
        else:
            raise ValueError("Mode not recognized. Choose on of 'equal_width', or 'equal_freq'.")
        self.n_bins = n_bins
        self.input_range = input_range

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def _fit_binary(self, X, y):
        if self.mode == 'equal_width':
            # Compute probability of class 1 in each equal width bin
            binned_stat = scipy.stats.binned_statistic(x=X[:, 1], values=np.equal(1, y), statistic='mean',
                                                       bins=self.n_bins, range=self.input_range)
            self.prob_class_1 = binned_stat.statistic  # TODO: test this and correct attributes
            self.binning = binned_stat.bin_edges
        elif self.mode == 'equal_freq':
            # Find binning based on equal frequency
            self.binning = np.quantile(X[:, 1],
                                       q=np.linspace(self.input_range[0], self.input_range[1], self.n_bins + 1))

            # Compute probability of class 1 in equal frequency bins
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = len(self.binning) - 1  # include rightmost edge in partition
            self.prob_class_1 = [y[digitized == i].mean() for i in range(1, len(self.binning))]

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binning", "prob_class_1"])
            # Find bin of predictions
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = len(self.binning) - 1  # include rightmost edge in partition
            # Transform to empirical frequency of class 1 in each bin
            p1 = np.array([self.prob_class_1[j] for j in (digitized - 1)])
            # If empirical frequency is NaN, do not change prediction
            p1 = np.where(np.isfinite(p1), p1, X[:, 1])
            assert np.all(np.isfinite(p1)), "Predictions are not all finite."

            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class BayesianBinningQuantiles(CalibrationMethod):
    """
    Probability calibration using Bayesian binning into quantiles

    Bayesian binning into quantiles [1]_ considers multiple equal frequency binning models and combines them through
    Bayesian model averaging. Each binning model :math:`M` is scored according to
    :math:`\\text{Score}(M) = P(M) \\cdot P(D | M),` where a uniform prior :math:`P(M)` is assumed. The marginal likelihood
    :math:`P(D | M)` has a closed form solution under the assumption of independent binomial class distributions in each
    bin with beta priors.

    Parameters
    ----------
        C : int, default = 10
            Constant controlling the number of binning models.
        input_range : list, shape (2,), default=[0, 1]
            Range of the scores to calibrate.

    .. [1] Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated Probabilities Using Bayesian Binning
           in Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, Austin, Texas, USA.
    """

    def __init__(self, C=10, input_range=[0, 1]):
        super().__init__()
        self.C = C
        self.input_range = input_range

    def _binning_model_logscore(self, probs, y, partition, N_prime=2):
        """
        Compute the log score of a binning model

        Each binning model :math:`M` is scored according to :math:`Score(M) = P(M) \\cdot P(D | M),` where a uniform prior
        :math:`P(M)` is assumed and the marginal likelihood :math:`P(D | M)` has a closed form solution
        under the assumption of a binomial class distribution in each bin with beta priors.

        Parameters
        ----------
        probs : array-like, shape (n_samples, )
            Predicted posterior probabilities.
        y : array-like, shape (n_samples, )
            Target classes.
        partition : array-like, shape (n_bins + 1, )
            Interval partition defining a binning.
        N_prime : int, default=2
            Equivalent sample size expressing the strength of the belief in the prior distribution.

        Returns
        -------
        log_score : float
            Log of Bayesian score for a given binning model
        """
        # Setup
        B = len(partition) - 1
        p = (partition[1:] - partition[:-1]) / 2 + partition[:-1]

        # Compute positive and negative samples in given bins
        N = np.histogram(probs, bins=partition)[0]

        digitized = np.digitize(probs, bins=partition)
        digitized[digitized == len(partition)] = len(partition) - 1  # include rightmost edge in partition
        m = [y[digitized == i].sum() for i in range(1, len(partition))]
        n = N - m

        # Compute the parameters of the Beta priors
        tiny = np.finfo(np.float).tiny  # Avoid scipy.special.gammaln(0), which can arise if bin has zero width
        alpha = N_prime / B * p
        alpha[alpha == 0] = tiny
        beta = N_prime / B * (1 - p)
        beta[beta == 0] = tiny

        # Prior for a given binning model (uniform)
        log_prior = - np.log(self.T)

        # Compute the marginal log-likelihood for the given binning model
        log_likelihood = np.sum(
            scipy.special.gammaln(N_prime / B) + scipy.special.gammaln(m + alpha) + scipy.special.gammaln(n + beta) - (
                    scipy.special.gammaln(N + N_prime / B) + scipy.special.gammaln(alpha) + scipy.special.gammaln(
                beta)))

        # Compute score for the given binning model
        log_score = log_prior + log_likelihood
        return log_score

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.binnings = []
            self.log_scores = []
            self.prob_class_1 = []
            self.T = 0
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
            return self

    def _fit_binary(self, X, y):
        # Determine number of bins
        N = len(y)
        min_bins = int(max(1, np.floor(N ** (1 / 3) / self.C)))
        max_bins = int(min(np.ceil(N / 5), np.ceil(self.C * N ** (1 / 3))))
        self.T = max_bins - min_bins + 1

        # Define (equal frequency) binning models and compute scores
        self.binnings = []
        self.log_scores = []
        self.prob_class_1 = []
        for i, n_bins in enumerate(range(min_bins, max_bins + 1)):
            # Compute binning from data and set outer edges to range
            binning_tmp = np.quantile(X[:, 1], q=np.linspace(self.input_range[0], self.input_range[1], n_bins + 1))
            binning_tmp[0] = self.input_range[0]
            binning_tmp[-1] = self.input_range[1]
            # Enforce monotonicity of binning (np.quantile does not guarantee monotonicity)
            self.binnings.append(np.maximum.accumulate(binning_tmp))
            # Compute score
            self.log_scores.append(self._binning_model_logscore(probs=X[:, 1], y=y, partition=self.binnings[i]))

            # Compute empirical accuracy for all bins
            digitized = np.digitize(X[:, 1], bins=self.binnings[i])
            # include rightmost edge in partition
            digitized[digitized == len(self.binnings[i])] = len(self.binnings[i]) - 1

            def empty_safe_bin_mean(a, empty_value):
                """
                Assign the bin mean to an empty bin. Corresponds to prior assumption of the underlying classifier
                being calibrated.
                """
                if a.size == 0:
                    return empty_value
                else:
                    return a.mean()

            self.prob_class_1.append(
                [empty_safe_bin_mean(y[digitized == k], empty_value=(self.binnings[i][k] + self.binnings[i][k - 1]) / 2)
                 for k in range(1, len(self.binnings[i]))])

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binnings", "log_scores", "prob_class_1", "T"])

            # Find bin for all binnings and the associated empirical accuracy
            posterior_prob_binnings = np.zeros(shape=[np.shape(X)[0], len(self.binnings)])
            for i, binning in enumerate(self.binnings):
                bin_ids = np.searchsorted(binning, X[:, 1])
                bin_ids = np.clip(bin_ids, a_min=0, a_max=len(binning) - 1)  # necessary if X is out of range
                posterior_prob_binnings[:, i] = [self.prob_class_1[i][j] for j in (bin_ids - 1)]

            # Computed score-weighted average
            norm_weights = np.exp(np.array(self.log_scores) - scipy.special.logsumexp(self.log_scores))
            posterior_prob = np.sum(posterior_prob_binnings * norm_weights, axis=1)

            # Compute probability for other class
            return np.column_stack([1 - posterior_prob, posterior_prob])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class GPCalibration(CalibrationMethod):
    """
    Probability calibration using a latent Gaussian process

    Gaussian process calibration [1]_ is a non-parametric approach to calibrate posterior probabilities from an arbitrary
    classifier based on a hold-out data set. Inference is performed using a sparse variational Gaussian process
    (SVGP) [2]_ implemented in `gpflow` [3]_.

    Parameters
    ----------
    n_classes : int
        Number of classes in calibration data.
    logits : bool, default=False
        Are the inputs for calibration logits (e.g. from a neural network)?
    mean_function : GPflow object
        Mean function of the latent GP.
    kernel : GPflow object
        Kernel function of the latent GP.
    likelihood : GPflow object
        Likelihood giving a prior on the class prediction.
    n_inducing_points : int, default=100
        Number of inducing points for the variational approximation.
    maxiter : int, default=1000
        Maximum number of iterations for the likelihood optimization procedure.
    n_monte_carlo : int, default=100
        Number of Monte Carlo samples for the inference procedure.
    max_samples_monte_carlo : int, default=10**7
        Maximum number of Monte Carlo samples to draw in one batch when predicting. Setting this value too large can
        cause memory issues.
    inf_mean_approx : bool, default=False
        If True, when inferring calibrated probabilities, only the mean of the latent Gaussian process is taken into
        account, not its covariance.
    session : tf.Session, default=None
        `tensorflow` session to use.
    random_state : int, default=0
        Random seed for reproducibility. Needed for Monte-Carlo sampling routine.
    verbose : bool
        Print information on optimization routine.

    References
    ----------
    .. [1] Wenger, J., Kjellström H. & Triebel, R. Non-Parametric Probabilistic Calibration for Classification
    .. [2] Hensman, J., Matthews, A. G. d. G. & Ghahramani, Z. Scalable Variational Gaussian Process Classification in
           Proceedings of AISTATS (2015)
    .. [3] Matthews, A. G. d. G., van der Wilk, M., et al. GPflow: A Gaussian process library using TensorFlow. Journal
           of Machine Learning Research 18, 1–6 (Apr. 2017)
    """

    def __init__(self,
                 n_classes,
                 logits=False,
                 mean_function=None,
                 kernel=None,
                 likelihood=None,
                 n_inducing_points=10,
                 maxiter=1000,
                 n_monte_carlo=100,
                 max_samples_monte_carlo=10 ** 7,
                 inf_mean_approx=False,
                 session=None,
                 random_state=1,
                 verbose=False):
        super().__init__()

        # Parameter initialization
        self.n_classes = n_classes
        self.verbose = verbose

        # Initialization of tensorflow session
        self.session = session

        # Initialization of Gaussian process components and inference parameters
        self.input_dim = n_classes
        self.n_monte_carlo = n_monte_carlo
        self.max_samples_monte_carlo = max_samples_monte_carlo
        self.n_inducing_points = n_inducing_points
        self.maxiter = maxiter
        self.inf_mean_approx = inf_mean_approx
        self.random_state = random_state
        np.random.seed(self.random_state)  # Set seed for optimization of hyperparameters

        with gpflow.defer_build():

            # Set likelihood
            if likelihood is None:
                self.likelihood = gp_classes.MultiCal(num_classes=self.n_classes,
                                                      num_monte_carlo_points=self.n_monte_carlo)
            else:
                self.likelihood = likelihood

            # Set mean function
            if mean_function is None:
                if logits:
                    self.mean_function = gpflow.conditionals.mean_functions.Identity()
                else:
                    self.mean_function = gp_classes.Log()
            else:
                self.mean_function = mean_function

            # Set kernel
            if kernel is None:
                k_white = gpflow.kernels.White(1, variance=0.01)
                if logits:
                    kernel_lengthscale = 10
                    k_rbf = gpflow.kernels.RBF(input_dim=1, lengthscales=kernel_lengthscale, variance=1)
                else:
                    kernel_lengthscale = 0.5
                    k_rbf = gpflow.kernels.RBF(input_dim=1, lengthscales=kernel_lengthscale, variance=1)
                    # Place constraints [a,b] on kernel parameters
                    k_rbf.lengthscales.transform = gpflow.transforms.Logistic(a=.001, b=10)
                    k_rbf.variance.transform = gpflow.transforms.Logistic(a=0, b=5)
                self.kernel = k_rbf + k_white
            else:
                self.kernel = kernel

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # Check for correct dimensions
        if X.ndim == 1 or np.shape(X)[1] != self.n_classes:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")

        # Create a new TF session if none is given
        if self.session is None:
            self.session = tf.Session(graph=tf.Graph())

        # Fit GP in TF session
        with self.session.as_default(), self.session.graph.as_default():
            if X.ndim == 1:
                raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
            else:
                self._fit_multiclass(X, y)
        return self

    def _fit_multiclass(self, X, y):
        # Setup
        y = y.reshape(-1, 1)

        # Select inducing points through scipy.cluster.vq.kmeans
        Z = scipy.cluster.vq.kmeans(obs=X.flatten().reshape(-1, 1),
                                    k_or_guess=min(X.shape[0] * X.shape[1], self.n_inducing_points, ))[0]

        # Define SVGP calibration model with multiclass softargmax calibration likelihood
        self.model = gp_classes.SVGPcal(X=X, Y=y, Z=Z,
                                        mean_function=self.mean_function,
                                        kern=self.kernel,
                                        likelihood=self.likelihood,
                                        whiten=True,
                                        q_diag=True)

        # Optimize parameters
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.model, maxiter=self.maxiter, disp=self.verbose)

        return self

    def predict_proba(self, X, mean_approx=False):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_classes)
            The uncalibrated posterior probabilities.
        mean_approx : bool, default=False
            If True, inference is performed using only the mean of the latent Gaussian process, not its covariance.
            Note, if `self.inference_mean_approximation==True`, then the logical value of this option is not considered.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, "model")

        if mean_approx or self.inf_mean_approx:

            # Evaluate latent GP
            with self.session.as_default(), self.session.graph.as_default():
                f, _ = self.model.predict_f(X_onedim=X.reshape(-1, 1))
                latent = f.eval().reshape(np.shape(X))

            # Return softargmax of fitted GP at input
            return scipy.special.softmax(latent, axis=1)
        else:

            with self.session.as_default(), self.session.graph.as_default():
                # Seed for Monte_Carlo
                tf.set_random_seed(self.random_state)

                if X.ndim == 1 or np.shape(X)[1] != self.n_classes:
                    raise ValueError("Calibration data must have shape (n_samples, n_classes).")
                else:
                    # Predict in batches to keep memory usage in Monte-Carlo sampling low
                    n_data = np.shape(X)[0]
                    samples_monte_carlo = self.n_classes * self.n_monte_carlo * n_data
                    if samples_monte_carlo >= self.max_samples_monte_carlo:
                        n_pred_batches = np.divmod(samples_monte_carlo, self.max_samples_monte_carlo)[0]
                    else:
                        n_pred_batches = 1

                    p_pred_list = []
                    for i in range(n_pred_batches):
                        if self.verbose:
                            print("Predicting batch {}/{}.".format(i + 1, n_pred_batches))
                        ind_range = np.arange(start=self.max_samples_monte_carlo * i,
                                              stop=np.minimum(self.max_samples_monte_carlo * (i + 1), n_data))
                        p_pred_list.append(tf.exp(self.model.predict_full_density(Xnew=X[ind_range, :])).eval())

                    return np.concatenate(p_pred_list, axis=0)

    def latent(self, z):
        """
        Evaluate the latent function f(z) of the GP calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence for which to evaluate the latent function.

        Returns
        -------
        f : array-like, shape=(n_evaluations,)
            Values of the latent function at z.
        f_var : array-like, shape=(n_evaluations,)
            Variance of the latent function at z.
        """
        # Evaluate latent GP
        with self.session.as_default(), self.session.graph.as_default():
            f, var = self.model.predict_f(z.reshape(-1, 1))
            latent = f.eval().flatten()
            latent_var = var.eval().flatten()

        return latent, latent_var

    def plot_latent(self, z, filename, plot_classes=True, **kwargs):
        """
        Plot the latent function of the calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence to plot latent function for.
        filename :
            Filename / -path where to save output.
        plot_classes : bool, default=True
            Should classes also be plotted?
        kwargs
            Additional arguments passed on to matplotlib.pyplot.subplots.

        Returns
        -------

        """
        # Evaluate latent GP
        with self.session.as_default(), self.session.graph.as_default():
            f, var = self.model.predict_f(z.reshape(-1, 1))
            latent = f.eval().flatten()
            latent_var = var.eval().flatten()
            Z = self.model.X.value

        # Plot latent GP
        if plot_classes:
            fig, axes = pycalib.texfig.subplots(nrows=2, ncols=1, sharex=True, **kwargs)
            axes[0].plot(z, latent, label="GP mean")
            axes[0].fill_between(z, latent - 2 * np.sqrt(latent_var), latent + 2 * np.sqrt(latent_var), alpha=.2)
            axes[0].set_ylabel("GP $g(\\bm{z}_k)$")
            axes[1].plot(Z.reshape((np.size(Z),)),
                         np.matlib.repmat(np.arange(0, self.n_classes), np.shape(Z)[0], 1).reshape((np.size(Z),)), 'kx',
                         markersize=5)
            axes[1].set_ylabel("class $k$")
            axes[1].set_xlabel("confidence $\\bm{z}_k$")
            fig.align_labels()
        else:
            fig, axes = pycalib.texfig.subplots(nrows=1, ncols=1, sharex=True, **kwargs)
            axes.plot(z, latent, label="GP mean")
            axes.fill_between(z, latent - 2 * np.sqrt(latent_var), latent + 2 * np.sqrt(latent_var), alpha=.2)
            axes.set_xlabel("GP $g(\\bm{z}_k)$")
            axes.set_ylabel("confidence $\\bm{z}_k$")

        # Save plot to file
        pycalib.texfig.savefig(filename)


class OneVsRestCalibrator(sklearn.base.BaseEstimator):
    """One-vs-the-rest (OvR) multiclass strategy
    Also known as one-vs-all, this strategy consists in fitting one calibrator
    per class. The probabilities to be calibrated of the other classes are summed.
    For each calibrator, the class is fitted against all the other classes.

    Parameters
    ----------
    calibrator : CalibrationMethod object
        A CalibrationMethod object implementing `fit` and `predict_proba`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    calibrators_ : list of `n_classes` estimators
        Estimators used for predictions.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    """

    def __init__(self, calibrator, n_jobs=None):
        self.calibrator = calibrator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Calibration data.
        y : (sparse) array-like, shape = [n_samples, ]
            Multi-class labels.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.calibrators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(OneVsRestCalibrator._fit_binary)(self.calibrator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i], self.label_binarizer_.classes_[i]]) for i, column in
            enumerate(columns))
        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, ["classes_", "calibrators_"])

        # Y[i, j] gives the probability that sample i has the label j.
        Y = np.array([c.predict_proba(
            np.column_stack([np.sum(np.delete(X, obj=i, axis=1), axis=1), X[:, self.classes_[i]]]))[:, 1] for i, c in
                      enumerate(self.calibrators_)]).T

        if len(self.calibrators_) == 1:
            # Only one estimator, but we still want to return probabilities for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        # Pad with zeros for classes not in training data
        if np.shape(Y)[1] != np.shape(X)[1]:
            p_pred = np.zeros(np.shape(X))
            p_pred[:, self.classes_] = Y
            Y = p_pred

        # Normalize probabilities to 1.
        Y = sklearn.preprocessing.normalize(Y, norm='l1', axis=1, copy=True, return_norm=False)
        return np.clip(Y, a_min=0, a_max=1)

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _first_calibrator(self):
        return self.calibrators_[0]

    @staticmethod
    def _fit_binary(calibrator, X, y, classes=None):
        """
        Fit a single binary calibrator.

        Parameters
        ----------
        calibrator
        X
        y
        classes

        Returns
        -------

        """
        # Sum probabilities of combined classes in calibration training data X
        cl = classes[1]
        X = np.column_stack([np.sum(np.delete(X, cl, axis=1), axis=1), X[:, cl]])

        # Check whether only one label is present in training data
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            if classes is not None:
                if y[0] == -1:
                    c = 0
                else:
                    c = y[0]
                warnings.warn("Label %s is present in all training examples." %
                              str(classes[c]))
            calibrator = _ConstantCalibrator().fit(X, unique_y)
        else:
            calibrator = clone(calibrator)
            calibrator.fit(X, y)
        return calibrator


class _ConstantCalibrator(CalibrationMethod):

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat([np.hstack([1 - self.y_, self.y_])], X.shape[0], axis=0)
