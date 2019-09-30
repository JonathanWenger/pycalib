# standard imports
import numpy as np
import tensorflow as tf
import gpflow

# Turn off tensorflow deprecation warnings
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

# gpflow imports
from gpflow.mean_functions import MeanFunction
from gpflow import features
from gpflow.conditionals import conditional, Kuu
from gpflow import settings
from gpflow.decors import params_as_tensors
from gpflow.quadrature import ndiag_mc
from gpflow.params import Parameter, DataHolder, Parameterized
from gpflow.models.model import GPModel
from gpflow import transforms, kullback_leiblers
from gpflow.models.svgp import Minibatch


############################
#    Mean Functions
############################

class Log(MeanFunction):
    """
    Natural logarithm prior mean function.

    :math:`y_i = \log(x_i)`
    """

    def __init__(self):
        MeanFunction.__init__(self)

    @params_as_tensors
    def __call__(self, X):
        # Avoid -inf = log(0)
        tiny = np.finfo(np.float).tiny
        X = tf.clip_by_value(X, clip_value_min=tiny, clip_value_max=np.inf)
        # Returns the natural logarithm of the input
        return tf.log(X)


class ScalarMult(MeanFunction):
    """
    Scalar multiplication mean function.

    :math:`y_i = \\alpha x_i`
    """

    def __init__(self, alpha=1):
        MeanFunction.__init__(self)
        self.alpha = Parameter(alpha, dtype=settings.float_type)

    @params_as_tensors
    def __call__(self, X):
        # Scalar multiplication
        return tf.multiply(self.alpha, X)


############################
#    Models
############################

class SVGPcal(gpflow.models.GPModel):
    """
    Probability calibration using a sparse variational latent Gaussian process.

    This is the Sparse Variational GP [1]_ calibration model. It has a single one-dimensional GP as a latent function
    which is applied to all inputs individually.

    .. [1] Hensman, J., Matthews, A. G. d. G. & Ghahramani, Z. Scalable Variational Gaussian Process Classification in
           Proceedings of AISTATS (2015)
    """

    def __init__(self, X, Y, kern, likelihood, feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, defaults to one.
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        if num_latent is None:
            num_latent = 1
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.num_classes = X.shape[1]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        # init variational parameters
        num_inducing = len(self.feature)

        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multi-output kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically referred to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing,
                                                                                     self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing,
                                                                                     self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL
        KL = self.build_prior_KL()

        # Get conditionals
        # TODO: allow for block-diagonal covariance
        fmeans, fvars = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations
        var_exp = self.likelihood.variational_expectations(fmeans, fvars, self.Y, full_cov=False)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Compute the mean and variance of :math:`p(f_* \\mid y)`.

        Parameters
        ----------
        Xnew : np.array, shape=(N, K)
        full_cov : bool
        full_output_cov : bool

        Returns
        -------
        mus, vars :
            Mean and covariances of the variational approximation to the GP applied to the K input dimensions of Xnew.
            Dimensions: mus= N x K and vars= N x K (x K)

        """
        # Reshape to obtain correct covariance
        num_data_new = tf.shape(Xnew)[0]
        Xnew = tf.reshape(Xnew, [-1, 1])

        # Compute conditional
        mu_tmp, var_tmp = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt,
                                      full_cov=full_cov,
                                      white=self.whiten, full_output_cov=full_output_cov)

        # Reshape to N x K
        mu = tf.reshape(mu_tmp + self.mean_function(Xnew), [num_data_new, self.num_classes])
        var = tf.reshape(var_tmp, [num_data_new, self.num_classes])

        return mu, var

    @params_as_tensors
    def predict_f(self, X_onedim, full_cov=False, full_output_cov=False):
        """
        Predict the one-dimensional latent function

        Parameters
        ----------
        X_onedim

        Returns
        -------

        """
        # Compute conditional
        mu, var = conditional(X_onedim, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt,
                              full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)

        return mu + self.mean_function(X_onedim), var

    @params_as_tensors
    def predict_full_density(self, Xnew):
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_full_density(pred_f_mean, pred_f_var)


############################
#    Inverse link functions
############################

class SoftArgMax(Parameterized):
    """
    This class implements the multi-class softargmax inverse-link function. Given a vector :math:`f=[f_1, f_2, ... f_k]`,
    then result of the mapping is :math:`y = [y_1 ... y_k]`, where
    :math:`y_i = \\frac{\\exp(f_i)}{\\sum_{j=1}^k\\exp(f_j)}`.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    @params_as_tensors
    def __call__(self, F):
        return tf.nn.softmax(F)


############################
#    Likelihoods
############################

class MultiCal(gpflow.likelihoods.Likelihood):
    def __init__(self, num_classes, invlink=None, num_monte_carlo_points=100, **kwargs):
        """
        A likelihood that performs multiclass calibration using the softargmax link function and a single latent
        process.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        invlink : default=None
            Inverse link function :math:`p(y \mid f)`.
        num_monte_carlo_points : int, default=100
            Number of Monte-Carlo points for prediction, i.e. for the integral :math:`\int p(y=Y|f)q(f) df`, where
            :math:`q(f)` is a Gaussian.
        kwargs
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if invlink is None:
            invlink = SoftArgMax(self.num_classes)
        elif not isinstance(invlink, SoftArgMax):
            raise NotImplementedError
        self.invlink = invlink
        self.num_monte_carlo_points = num_monte_carlo_points

    def logp(self, F, Y):
        """
        Computes the log softargmax at indices from Y.

        :math:`\\sigma(F)_y = \\frac{exp(F_y)}{\\sum_{k=1}^K \\exp(F_k)}`

        Parameters
        ----------
        F : tf.tensor, shape=(N, K)
            Inputs to softargmax.
        Y : tf.tensor, shape=(N, 1)
            Indices of softargmax output.

        Returns
        -------
        log_sigma_y : tf.tensor, shape=()
            log softargmax at y

        """
        if isinstance(self.invlink, SoftArgMax):
            with tf.control_dependencies(
                    [
                        tf.assert_equal(tf.shape(Y)[1], 1),
                        tf.assert_equal(tf.cast(tf.shape(F)[1], settings.int_type),
                                        tf.cast(self.num_classes, settings.int_type))
                    ]):
                return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:, 0])[:, None]
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmus, Fvars, Y, full_cov=False):
        """
        Computes an approximation to the expectation terms in the variational objective.

        This function approximates the :math:`n` expectation terms :math:`\\mathbb{E}_{q(f_n)}[\\log p(y_n \\mid f_n)]`
        in the variational objective function.

        Parameters
        ----------
        Fmus : tf.tensor, shape=(N, K)
            Means of the latent GP at input locations X. Dimension N x K.
        Fvars : tf.tensor, shape=(N, K(, K))
            Variances of the latent GP at input locations X. Dimension N x K (x K).
        Y : tf.tensor, shape=(N,)
            Output vector.

        Returns
        -------
        ve : tf.tensor, shape=(N,)
            The variational expectation assuming a Gaussian approximation q.
        """
        if isinstance(self.invlink, SoftArgMax):
            # Compute variational expectations by 2nd order Taylor approximation
            sigma_mu = tf.nn.softmax(Fmus, axis=1)

            if full_cov:
                sigSsig = tf.einsum("nk, nkl, nl -> n", sigma_mu, Fvars, sigma_mu)
            else:
                sigSsig = tf.reduce_sum(tf.multiply(tf.multiply(sigma_mu, Fvars), sigma_mu), axis=1)

            diagSsig = tf.reduce_sum(tf.multiply(sigma_mu, Fvars), axis=1)
            logsoftargmax_y = tf.squeeze(self.logp(Fmus, Y))

            # Objective function
            return logsoftargmax_y + 0.5 * (sigSsig - diagSsig)

        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmus, Fvars):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2

        Parameters
        ----------
        Fmus : array/tensor, shape=(N, D)
            Mean(s) of Gaussian density.
        Fvars : array/tensor, shape=(N, D(, D))
            Covariance(s) of Gaussian density.
        """
        raise NotImplementedError

    def predict_density(self, Fmus, Fvars, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y, compute the log predictive density of Y.
        i.e. if :math:`p(f_* | y) = \\mathcal{N}(Fmu, Fvar)` and :math:`p(y_*|f_*)` is the likelihood, then this
        method computes the log predictive density :math:`\\log \\int p(y_*|f)p(f_* | y) df`. Here, we implement a
        Monte-Carlo routine.

        Parameters
        ----------
        Fmus : array/tensor, shape=(N, K)
            Mean(s) of Gaussian density.
        Fvars : array/tensor, shape=(N, K(, K))
            Covariance(s) of Gaussian density.
        Y : arrays/tensors, shape=(N(, K))
            Deterministic arguments to be passed by name to funcs.

        Returns
        -------
        log_density : array/tensor, shape=(N(, K))
            Log predictive density.
        """
        if isinstance(self.invlink, SoftArgMax):
            return ndiag_mc(self.logp, self.num_monte_carlo_points, Fmus, Fvars,
                            logspace=True, epsilon=None, Y=Y)
        else:
            raise NotImplementedError

    def predict_full_density(self, Fmus, Fvars):
        if isinstance(self.invlink, SoftArgMax):
            # Sample from standard normal
            N = tf.shape(Fmus)[0]
            epsilon = tf.random_normal((self.num_monte_carlo_points, N, self.num_classes),
                                       dtype=settings.float_type)

            # Transform to correct mean and covariance
            f_star = Fmus[None, :, :] + tf.sqrt(Fvars[None, :, :]) * epsilon  # S x N x K

            # Compute Softmax
            p_y_f_star = tf.nn.softmax(f_star, axis=2)

            # Average to obtain log Monte-Carlo estimate
            return tf.log(tf.reduce_mean(p_y_f_star, axis=0))
        else:
            raise NotImplementedError