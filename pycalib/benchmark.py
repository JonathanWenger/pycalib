# Standard library imports
import os
import time
import glob
import re
import h5py

# Numerics and plotting
import numpy as np
import pandas as pd
import scipy.special
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from collections import OrderedDict

# scikit-learn imports
import sklearn.utils
import sklearn.model_selection as ms
from sklearn.datasets import fetch_openml

# pytorch imports
import torch
import torch.utils.data
import pretrainedmodels
import pretrainedmodels.utils

# Package imports
import pycalib.calibration_methods as calm
import pycalib.texfig as texfig
import pycalib.scoring as sc
import pycalib.plotting


class Benchmark(object):
    """
    A benchmarking class for calibration methods.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    cross_validator : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - `CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multi-class, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, cal_methods, cal_method_names, cross_validator, random_state=None):
        # Create directory
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        # Check models for consistency
        if not isinstance(cal_methods, (tuple, list)):
            cal_methods = [cal_methods]
        if not isinstance(cal_method_names, (tuple, list)):
            cal_method_names = [cal_method_names]
        for m in cal_methods:
            assert isinstance(m, calm.CalibrationMethod), "Method is not a subclass of CalibrationMethod."
        self.cal_methods = cal_methods
        self.cal_method_names = cal_method_names
        assert len(self.cal_methods) == len(self.cal_method_names)
        # Set cross validator and random state
        self.cross_validator = cross_validator
        self.train_size = cross_validator.train_size
        self.test_size = cross_validator.test_size
        self.random_state = random_state
        # Initialize results
        self.results = None

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def run(self, n_jobs=None):
        """
        Train all models, evaluate on test data and save the results.

        Parameters
        ----------
        n_jobs : int or None, optional (default=None)
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        """
        # Initialization
        scores = pd.DataFrame()
        start_time = time.time()

        # Iterate through models
        for cal_method, cal_method_name in zip(self.cal_methods, self.cal_method_names):

            # Create cal_method and plot directory
            method_dir = os.path.join(self.run_dir, "methods", cal_method_name)
            os.makedirs(method_dir, exist_ok=True)
            plot_dir = os.path.join(method_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            # Initialize cal_method specific scores data frame
            model_scores = pd.DataFrame()
            m, s = divmod(time.time() - start_time, 60)
            h, m = divmod(m, 60)
            print("\n\n=== Model {}: {}/{}, Total Time: {:.0f}:{:02.0f}:{:02.0f} ===\n".format(cal_method_name,
                                                                                               self.cal_method_names.index(
                                                                                                   cal_method_name) + 1,
                                                                                               len(self.cal_methods), h, m,
                                                                                               s))
            print("Running {} {}-fold CV in {}".format(cal_method_name, self.cross_validator.n_splits, method_dir))

            # Reset random state of cross validator to guarantee same splits for each cal_method
            self.cross_validator.random_state = sklearn.utils.check_random_state(self.random_state)

            # Iterate through data sets
            for X, y, data_params in self.data_gen():

                # Create directories and measure time
                dataset_time = time.time()
                if "Model" in data_params:
                    plot_model_dir = os.path.join(plot_dir, data_params["Model"])
                    os.makedirs(plot_model_dir, exist_ok=True)
                else:
                    plot_model_dir = plot_dir

                # Define scorers and parameters
                labels = np.sort(np.unique(y))
                if len(np.unique(y)) < np.shape(X)[1]:
                    labels = np.array(range(0, np.shape(X)[1]))

                metrics_dict = {
                    "accuracy": (sc.accuracy, {}),
                    "log-loss": (sklearn.metrics.log_loss, {"labels": labels}),
                    "ECE": (sc.expected_calibration_error, {"p": 1}),
                    "ECE_15bins": (sc.expected_calibration_error, {"p": 1, "n_bins": 15}),
                    "MCE": (sc.expected_calibration_error, {"p": np.inf}),
                    "sharpness": (sc.sharpness, {}),
                    "overconfidence": (sc.overconfidence, {}),
                    "underconfidence": (sc.underconfidence, {}),
                    "avg_confidence": (sc.average_confidence, {}),
                    "weighted_abs_conf_difference": (sc.weighted_abs_conf_difference, {})
                }
                if len(labels) == 2:
                    metrics_dict["brier_score"] = (sc.brier_score, {})

                scorer = sc.MultiScorer(
                    metrics=metrics_dict,
                    plots={
                        "reliability_diagram": (pycalib.plotting.reliability_diagram,
                                                {"filename": os.path.join(plot_model_dir,
                                                                          "reliability_diagram"),
                                                 "show_ece": True, "title": None, })
                    })

                # TODO: add timing methodology for cross validation
                # Run cross validation for current cal_method specification
                ms.cross_val_score(estimator=cal_method,
                                   X=X, y=y,
                                   scoring=scorer,
                                   cv=self.cross_validator,
                                   n_jobs=n_jobs,
                                   pre_dispatch="1*n_jobs",
                                   verbose=0)

                # TODO: right now this prevents n_jobs != 1
                model_scores_dict = scorer.get_results(fold="all")

                # Convert result to data frame and add to cal_method scores data frame
                model_scores_tmp = pd.DataFrame(model_scores_dict)
                info_params = {**{"Method": cal_method_name}, **data_params}
                model_scores_tmp = model_scores_tmp.assign(**info_params)
                model_scores = model_scores.append(model_scores_tmp)

                # Save summary statistics
                grouped = model_scores_tmp.groupby(list(info_params.keys()))
                summ_stats_model_scores = grouped.agg([np.mean, np.std])
                scores = scores.append(summ_stats_model_scores)

                # Log summary statistics for current cal_method and data set
                m, s = divmod(time.time() - dataset_time, 60)
                h, m = divmod(m, 60)
                print("{} {}-fold results:\t ECE: {:.4f} +/- {:.4f}\t Time: {:.0f}:{:02.0f}:{:02.0f}".format(
                    cal_method_name,
                    self.cross_validator.n_splits,
                    np.mean(model_scores_dict["ECE"]),
                    np.std(model_scores_dict["ECE"]),
                    h, m, s
                ))

            # Save cal_method specific CV results to file
            tmp_time = time.strftime("%Y%m%d-%Hh%Mm%Ss")
            model_scores.to_csv(path_or_buf=os.path.join(method_dir, "cv_scores_{}_{}.csv".format(cal_method_name, tmp_time)))
            # Save summary statistics
            mi = model_scores.columns
            model_scores.columns = pd.Index([e[0] + "_" + e[1] for e in mi.tolist()])
            scores.to_csv(path_or_buf=os.path.join(method_dir, "cv_scores_{}_{}_summ.csv".format(cal_method_name, tmp_time)))

        # Save summary statistics of all models to file
        mi = scores.columns
        scores.columns = pd.Index([e[0] + "_" + e[1] for e in mi.tolist()])
        scores.to_csv(
            path_or_buf=os.path.join(self.run_dir, "cv_scores_total_{}.csv".format(time.strftime("%Y%m%d-%Hh%Mm%Ss"))))

        self.results = scores

    @staticmethod
    def plot(out_file, results_file, score, methods, classifiers="all", width=5., height=2.5):
        """
        Plot results from benchmark experiments as an error bar plot.

        Parameters
        ----------
        out_file : str
            File location for the output plot.
        results_file : str
             The location of the csv files containing experiment results.
        score : str
            Type of score to plot.
        methods : list
            Calibration methods to plot.
        classifiers : list or "all"
            List of classifiers for which to show results.
        width : float, default=5.
            Width of the plot.
        height : float, default=2.5
            Height of the plot.
        """
        # Load data from file
        results = pd.read_csv(results_file)

        # Filter data
        if isinstance(classifiers, str) and classifiers == "all":
            classifiers = pd.unique(results["Model"])

        # Colors
        cmap = get_cmap("tab10")  # https://matplotlib.org/gallery/color/colormap_reference.html

        # Create figure
        fig, axes = texfig.subplots(width=width, ratio=height/width)
        x_base = np.arange(len(classifiers)) + 1
        x = np.array([x_base - .2, x_base, x_base + .2])

        for i_method, method in enumerate(methods):
            for i_clf, clf in enumerate(classifiers):
                results_tmp = results.loc[(results["Method"] == method) & (results["Model"] == clf)]
                ymean = results_tmp[score + "_mean"]
                y2sigma = 2 * results_tmp[score + "_std"]
                marker, _, bar = axes.errorbar(x[i_method, i_clf], y=ymean, yerr=y2sigma,
                                               color=cmap.colors[i_method], ecolor=cmap.colors[i_method],
                                               fmt='o', elinewidth=2, capsize=0, label=method)
                bar[0].set_alpha(0.75)

        # axes.set_xlabel("model")
        plt.xticks(x_base)
        axes.set_xticklabels(classifiers, rotation=45, ha="right")
        axes.set_ylabel(score)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        # Save figure
        texfig.savefig(out_file)
        plt.close('all')


class SyntheticBeta(Benchmark):
    """
    Model evaluation using synthetic data sampled from a Beta distribution.

    Implements a data generation method returning a new evaluation data with maximum posterior probabilities sampled
    from a Beta distribution :math:`\hat{p}_{\max} \sim (1-\\frac{1}{K})\\text{Beta}(\\alpha, \\beta)+\\frac{1}{K}` and
    corresponding class labels sampled from a Bernoulli distribution with parameter :math:`f(\\hat{p}_{\max})`, where
    :math:`f : [\\frac{1}{n_\\text{classes}},1] \\rightarrow [\\frac{1}{n_\\text{classes}},1]` is a miscalibration function.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    beta_params : tuple or list, shape=(n,2)
        Parameters :math:`(\\alpha, \\beta)` of the Beta distribution.
    miscal_functions : function or list
        Function(s) :math:`f : [0,1] \\rightarrow [0,1]` for miscalibration. When this function is different from the
        identity, the generated output from this function is miscalibrated. The function automatically gets rescaled to
        :math:`f : [\\frac{1}{n_\\text{classes}},1] \\rightarrow [\\frac{1}{n_\\text{classes}},1]`.
    miscal_function_names : str or list
        Names of miscalibration functions.
    size : int or list
        Size of data set.
    marginal_probs : float or list, default=None
        Marginal class probabilities.
    n_splits : int, default=10
        Number of splits for cross validation.
    test_size : float, default=0.9
        Size of test set.
    train_size : float, default=None
        Size of calibration set.
    random_state : int, RandomState instance or None, optional (default=None)
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, cal_methods, cal_method_names,
                 beta_params, miscal_functions, miscal_function_names, size, marginal_probs=None,
                 n_splits=10, test_size=0.9, train_size=None, random_state=None):

        # Instantiate cross validator
        cv = ms.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                             random_state=sklearn.utils.check_random_state(random_state))

        # Run super class constructor
        super().__init__(run_dir=run_dir, cal_methods=cal_methods, cal_method_names=cal_method_names, cross_validator=cv,
                         random_state=random_state)

        # Check synthetic data parameters
        if not isinstance(beta_params, (tuple, list, np.ndarray)):
            beta_params = [beta_params]
        if not isinstance(miscal_functions, (tuple, list)):
            miscal_functions = [miscal_functions]
        if not isinstance(miscal_function_names, (tuple, list)):
            miscal_function_names = [miscal_function_names]
        assert len(miscal_functions) == len(
            miscal_function_names), "Lists of miscalibration functions and their names have different sizes."
        if not isinstance(size, (tuple, list, np.ndarray)):
            size = [size]
        if not isinstance(marginal_probs, (tuple, list)):
            n_classes = np.shape(marginal_probs)[0]
            assert n_classes > 1, "Must have more than one class."
            marginal_probs = [marginal_probs]
        else:
            n_classes = np.shape(marginal_probs[0])[0]
            assert np.all([np.shape(mp)[0] == n_classes for mp in
                           marginal_probs]), "Marginal probabilities must have the same shape."
        self.n_classes = n_classes
        self.beta_params = beta_params
        self.miscal_functions = miscal_functions
        self.miscal_function_names = miscal_function_names
        self.size = size
        self.marginal_probs = marginal_probs

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """

        # Automatically rescale miscalibration function to :math:`f : [1/n_classes,1] \\rightarrow [1/n_classes,1]`
        def mf(f, x):
            y = (x - 1 / self.n_classes) / (1 - 1 / self.n_classes)
            w = f(y)
            return w * (1 - 1 / self.n_classes) + 1 / self.n_classes

        # Return data with all parameter combinations
        generator_comprehension = (
            SyntheticBeta.sample_miscal_data(alpha=bp[0], beta=bp[1],
                                             miscal_func=lambda x: mf(f=self.miscal_functions[i], x=x),
                                             miscal_func_name=self.miscal_function_names[i], size=s,
                                             marginal_probs=marg, random_state=self.random_state)
            for bp in self.beta_params
            for i, _ in enumerate(self.miscal_functions)
            for s in self.size
            for marg in self.marginal_probs)
        return generator_comprehension

    @staticmethod
    def sample_miscal_data(alpha, beta, miscal_func, miscal_func_name, size, marginal_probs,
                           random_state=None):
        """
        Sample a synthetic data set based on the Beta distribution and a miscalibration function.

        Parameters
        ----------
        alpha : float
            Parameter :math:`\alpha` of the Beta distribution.
        beta : float
            Parameter :math:`\alpha` of the Beta distribution.
        miscal_func : function
            Function :math:`f : [\\frac{1}{n_\\text{classes}},1] \\rightarrow [\\frac{1}{n_\\text{classes}},1]` giving
            the accuracy for a given confidence. When this function is different from the identity, the generated output
            from this function is miscalibrated.
        miscal_func_name : str
            Name of the miscalibration function.
        size : int
            Size of data set.
        marginal_probs : np.ndarray, size=(n_classes,)
            Marginal class probabilities.
        random_state : int, RandomState instance or None, optional (default=None)
            If `int`, `random_state` is the seed used by the random number generator;
            If `RandomState` instance, `random_state` is the random number generator;
            If `None`, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Information on sampled data
        info_dict = {"alpha": alpha,
                     "beta": beta,
                     "miscal_function": miscal_func_name,
                     "test_plus_train_size": size,
                     "marginal_class_probs": ', '.join(str(e) for e in marginal_probs)}
        n_classes = len(marginal_probs)

        # Sample maximum posterior probabilities from a Beta distribution
        random_state = sklearn.utils.check_random_state(random_state)
        p_hat = random_state.beta(a=alpha, b=beta, size=size) * (1 - 1 / n_classes) + 1 / n_classes

        # Sample true class labels
        y = random_state.choice(a=range(0, n_classes), size=size, replace=True, p=marginal_probs)

        # Sample predicted probabilities with the correctness according to the miscalibration function
        p_pred = np.zeros([size, n_classes])
        for i, p in enumerate(p_hat):
            p_pred[i, :] = SyntheticBeta._sample_probs(y_true=y[i], p_max=p, p_miscal=miscal_func(p),
                                                       n_classes=n_classes,
                                                       random_state=random_state)

        return p_pred, y, info_dict

    @staticmethod
    def _sample_probs(y_true, p_max, p_miscal, n_classes, gamma=2, random_state=None, tol=10 ** -9):
        """
        Samples a probability vector such that the given class is predicted with the given probability.

        Parameters
        ----------
        y_true : int
            Given class prediction.
        p_max : float
            Predicted probability.
        p_miscal : float
            Miscalibrated probability of the correct class prediction.
        n_classes : int
            Number of classes.
        gamma : float, default=2
            Parameter in stick-breaking process view. The smaller gamma, the more concentrated the sampled probabilities
            will be.
        random_state : int, RandomState instance or None, optional (default=None)
            If `int`, `random_state` is the seed used by the random number generator;
            If `RandomState` instance, `random_state` is the random number generator;
            If `None`, the random number generator is the RandomState instance used
            by `np.random`.
        tol : float, default=10^-9
            Tolerance parameter voiding numerical issues. Returned probabilities satisfy
            `np.abs(1 - np.sum(p_pred)) < tol`.

        Returns
        -------
        p_pred : np.ndarray
            Array of posterior probabilities.

        """
        # Initialization
        random_state = sklearn.utils.check_random_state(random_state)
        upper_bound = np.minimum(1 - p_max, p_max)
        lower_bound = np.maximum(1 - p_max - (n_classes - 2) * p_max, 0)
        assert gamma > 0, "Gamma must be positive."
        assert upper_bound >= lower_bound, "Given p_max cannot be maximal for the number of classes."
        res_probs = np.zeros(n_classes)
        res_probs[0] = p_max
        for k in range(1, n_classes):
            # Sample class probabilities uniformly given that p_max is the largest value
            if upper_bound <= 0:
                res_probs[k] = 0
            if upper_bound == lower_bound:
                res_probs[k] = upper_bound
            else:
                # Sample from beta distribution, can be seen as a stick-breaking process with length constraints
                res_probs[k] = np.clip((upper_bound - lower_bound) * random_state.beta(a=1, b=gamma, size=1)
                                       + lower_bound, a_min=lower_bound, a_max=upper_bound)
            assert res_probs[k] <= p_max
            # Update lower and upper bound, this is determined by p <= p_max and the number of classes
            upper_bound = np.minimum(1 - np.sum(res_probs[0:(k + 1)]), p_max)
            upper_bound = np.maximum(upper_bound, 0)
            lower_bound = np.maximum((1 - np.sum(res_probs[0:(k + 1)]) - (n_classes - k - 2) * p_max), 0)
            lower_bound = np.minimum(lower_bound, upper_bound)

        if np.abs(1 - np.sum(res_probs)) > tol:
            # res_probs /= np.sum(res_probs)
            raise ValueError("Probabilities do not sum to 1.")
        assert np.argmax(res_probs) == 0, "Given supposedly maximal probability is not maximal."
        assert np.abs(p_max - np.max(res_probs)) < 10 ** -9

        # Reorder probabilities to ensure p_max matches predicted class
        other_class = random_state.choice([k for k in range(0, n_classes) if k != y_true], size=1).item()
        class_pred = random_state.choice(a=[y_true, other_class], p=[p_miscal, 1 - p_miscal], size=1).item()

        # Mix probabilities other than p_max randomly to remove dependence introduced through the bounds
        p_pred = random_state.choice(res_probs[1:n_classes], replace=False, size=n_classes - 1)

        # Insert p_max to match predicted class
        p_pred = np.insert(p_pred, class_pred, p_max)
        assert p_pred[class_pred] == p_max, "Maximal probability does not match predicted class."
        assert np.all(p_pred >= 0), "Probabilities are not non-negative."
        assert np.abs(1 - np.sum(p_pred)) < 10 ** -9, "Probabilities do not sum to 1."

        return p_pred

    def plot(self, **kwargs):
        """
        Plots the result of the benchmark experiment.

        Parameters
        ----------
        **kwargs :
            Additional arguments passed on to :func:`matplotlib.plot`.

        """
        metric = "underconfidence"

        # Read data from file
        df = pd.read_csv(os.path.join(self.run_dir, "cv_scores.csv"))
        plot_dir = os.path.join(self.run_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Find parameter combinations and select data
        param_columns = ["alpha", "beta", "miscal_function", "test_plus_train_size", "marginal_class_probs"]
        param_combinations = df.loc[:, param_columns].drop_duplicates()

        for i in range(param_combinations.shape[0]):
            # Select data with specific parameter combination
            ind = np.all(np.array([df[c] == param_combinations.loc[i, c] for c in param_columns]), axis=0)
            df_tmp = df.loc[ind, :]

            # Plot
            fig = texfig.figure()
            plt.errorbar(x=df_tmp["model"],
                         y=np.abs(df_tmp["test_" + metric + "_mean"]),
                         yerr=df_tmp["test_" + metric + "_std"],
                         fmt='o')
            plt.xticks(rotation=45, ha="right")
            plt.ylabel(metric)

            texfig.savefig(
                filename=os.path.join(plot_dir, metric + "_" + '_'.join(str(e) for e in param_combinations.loc[i, :])))

    def plot_miscal_function(self, function_names=None, **kwargs):
        """
        Plots the miscalibration functions.

        Parameters
        ----------
        function_names : list
            List of miscalibration functions to plot.
        **kwargs :
            Additional arguments passed on to :func:`matplotlib.plot`.

        """
        # Setup
        xx = np.linspace(0.5, 1, 1000)
        fig, ax = plt.subplots()
        ax.plot(xx, xx, linestyle='dashed', color="gray")
        if function_names is None:
            function_names = self.miscal_function_names
        f_ind = [i for i, name in enumerate(self.miscal_function_names) if name in function_names]

        # Plot miscalibration functions
        for name_f, f in zip(self.miscal_function_names[f_ind], self.miscal_functions[f_ind]):
            ax.plot(xx, f(xx), label=name_f, **kwargs)
        ax.legend()
        ax.set_xlabel('predicted probability')
        ax.set_ylabel('empirical accuracy')

        return ax


class KITTIBinaryData(Benchmark):
    """
    Model evaluation using the (binary) KITTI benchmark dataset.

    This class implements a data generation method returning a new evaluation data set for each scoring round.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    data_dir : str
        Directory containing calibration data obtained from KITTI classification.
    classifier_names : list
        Names of classifiers to be calibrated. Classification results on KITTI must be contained in `data_dir`.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    n_splits : int, default=10
        Number of splits for cross validation.
    test_size : float, default=0.9
        Size of test set.
    train_size : float, default=None
        Size of calibration set.
    random_state : int, RandomState instance or None, optional (default=None)
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, data_dir, classifier_names, cal_methods, cal_method_names,
                 n_splits=10, test_size=0.9, train_size=None, random_state=None):
        # Create cross validator which splits the data randomly based on test or train size
        cv = ms.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                             random_state=sklearn.utils.check_random_state(random_state))
        self.data_dir = data_dir
        self.classifier_names = classifier_names

        # Run super class constructor
        super().__init__(run_dir=run_dir, cal_methods=cal_methods, cal_method_names=cal_method_names, cross_validator=cv,
                         random_state=random_state)

    @staticmethod
    def classify_val_data(file, clf_name, classifier, data_folder='kitti', output_folder='clf_output'):
        """
        Classify the (binary) KITTI evaluation data set with a given model.

        Parameters
        ----------
        file : str
            Output from the model is saved in the given directory.
        clf_name :
            Name of classifier.
        classifier : sklearn.base.BaseEstimator
            Classifier to classify KITTI data with.
        data_folder : str, default=''
            Folder where data is stored.
        output_folder : str, default='clf_output'
            Name of folder where output is to be stored. This folder is created if non-existent.

        Returns
        -------

        """
        # Load data from file
        files = ['kitti_all_train.data',
                 'kitti_all_train.labels',
                 'kitti_all_test.data',
                 'kitti_all_test.labels']
        X_train = np.loadtxt(os.path.join(file, data_folder, files[0]), np.float64, skiprows=1)
        y_train = np.loadtxt(os.path.join(file, data_folder, files[1]), np.int32, skiprows=1)
        X_test = np.loadtxt(os.path.join(file, data_folder, files[2]), np.float64, skiprows=1)
        y_test = np.loadtxt(os.path.join(file, data_folder, files[3]), np.int32, skiprows=1)

        y_train = np.where(y_train > 0, 1, 0)
        y_test = np.where(y_test > 0, 1, 0)

        # Train classifier
        classifier.fit(X=X_train, y=y_train)

        # Classify validation data
        p_pred = classifier.predict_proba(X_test)
        y_y_pred = np.column_stack([y_test, np.argmax(p_pred, axis=1)])

        # Remove possible nan or inf
        isnotnan_ind = ~np.isnan(p_pred).any(axis=1)
        p_pred = p_pred[isnotnan_ind, :]
        y_y_pred = y_y_pred[isnotnan_ind, :]

        # Save to file
        os.makedirs(file, exist_ok=True)
        out_path = os.path.join(file, output_folder)
        os.makedirs(out_path, exist_ok=True)
        np.savetxt(os.path.join(out_path, "p_pred_{}.csv".format(clf_name)), p_pred, delimiter=",")
        np.savetxt(os.path.join(out_path, "y_y_pred_{}.csv".format(clf_name)), y_y_pred, fmt='%i', delimiter=",")

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Load logits and classification results from file
        for clf_name in self.classifier_names:
            X = np.loadtxt(os.path.join(self.data_dir, 'p_pred_' + clf_name + '.csv'), dtype=float, delimiter=',')
            y = np.loadtxt(os.path.join(self.data_dir, 'y_y_pred_' + clf_name + '.csv'),
                           dtype=int, delimiter=',')[:, 0]

            # Calibration and test size
            if self.test_size > 1:
                calib_size_tmp = self.train_size
                test_size_tmp = self.test_size
            else:
                calib_size_tmp = int(np.shape(X)[0] * self.train_size)
                test_size_tmp = np.shape(X)[0] - int(np.shape(X)[0] * self.train_size)

            # Collect information on data set
            info_dict = {
                "Dataset": "KITTI",
                "Model": clf_name,
                "n_classes": np.shape(X)[1],
                "size": np.shape(X)[0],
                "calib_size": calib_size_tmp,
                "test_size": test_size_tmp}

            # Return (as generator)
            yield X, y, info_dict


class PCamData(Benchmark):
    """
    Model evaluation using the PCam benchmark dataset.

    This class implements a data generation method returning a new evaluation data set for each scoring round.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    data_dir : str
        Directory containing calibration data obtained from PCam classification.
    classifier_names : list
        Names of classifiers to be calibrated. Classification results on PCam must be contained in `data_dir`.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    n_splits : int, default=10
        Number of splits for cross validation.
    test_size : float, default=0.9
        Size of test set.
    train_size : float, default=None
        Size of calibration set.
    random_state : int, RandomState instance or None, optional (default=None)
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, data_dir, classifier_names, cal_methods, cal_method_names,
                 n_splits=10, test_size=0.9, train_size=None, random_state=None):
        # Create cross validator which splits the data randomly based on test or train size
        cv = ms.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                             random_state=sklearn.utils.check_random_state(random_state))
        self.data_dir = data_dir
        self.classifier_names = classifier_names

        # Run super class constructor
        super().__init__(run_dir=run_dir, cal_methods=cal_methods, cal_method_names=cal_method_names, cross_validator=cv,
                         random_state=random_state)

    @staticmethod
    def classify_val_data(file, clf_name, classifier, data_folder='pcam', output_folder='clf_output'):
        """
        Classify the PCam evaluation data set with a given model.

        Parameters
        ----------
        file : str
            Output from the model is saved in the given directory.
        clf_name :
            Name of classifier.
        classifier : sklearn.base.BaseEstimator
            Classifier to classify p53 data with.
        data_folder : str, default=''
            Folder where data is stored.
        output_folder : str, default='clf_output'
            Name of folder where output is to be stored. This folder is created if non-existent.

        Returns
        -------

        """
        # Load data from file
        with h5py.File(os.path.join(file, "camelyonpatch_level_2_split_valid_x.h5"), 'r') as hf:
            X = hf["x"][:]

        # Convert to grayscale by weighted average (luminosity method)
        X = np.dot(X[..., :3], [0.299, 0.587, 0.114])

        # Reshape and normalize
        X = X.reshape(np.shape(X)[0], -1) / 255

        with h5py.File(os.path.join(file, "camelyonpatch_level_2_split_valid_y.h5"), 'r') as hf:
            y = hf["y"][:]
        y = y.flatten()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=10000,
                                                                                    random_state=42)

        # Train classifier
        classifier.fit(X=X_train, y=y_train)

        # Classify validation data
        p_pred = classifier.predict_proba(X_test)
        y_y_pred = np.column_stack([y_test, np.argmax(p_pred, axis=1)])

        # Remove possible nan or inf
        isnotnan_ind = ~np.isnan(p_pred).any(axis=1)
        p_pred = p_pred[isnotnan_ind, :]
        y_y_pred = y_y_pred[isnotnan_ind, :]

        # Save to file
        os.makedirs(file, exist_ok=True)
        out_path = os.path.join(file, output_folder)
        os.makedirs(out_path, exist_ok=True)
        np.savetxt(os.path.join(out_path, "p_pred_{}.csv".format(clf_name)), p_pred, delimiter=",")
        np.savetxt(os.path.join(out_path, "y_y_pred_{}.csv".format(clf_name)), y_y_pred, fmt='%i', delimiter=",")

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Load logits and classification results from file
        for clf_name in self.classifier_names:
            X = np.loadtxt(os.path.join(self.data_dir, 'p_pred_' + clf_name + '.csv'), dtype=float, delimiter=',')
            y = np.loadtxt(os.path.join(self.data_dir, 'y_y_pred_' + clf_name + '.csv'),
                           dtype=int, delimiter=',')[:, 0]

            # Calibration and test size
            if self.test_size > 1:
                calib_size_tmp = self.train_size
                test_size_tmp = self.test_size
            else:
                calib_size_tmp = int(np.shape(X)[0] * self.train_size)
                test_size_tmp = np.shape(X)[0] - int(np.shape(X)[0] * self.train_size)

            # Collect information on data set
            info_dict = {
                "Dataset": "PCam",
                "Model": clf_name,
                "n_classes": np.shape(X)[1],
                "size": np.shape(X)[0],
                "calib_size": calib_size_tmp,
                "test_size": test_size_tmp}

            # Return (as generator)
            yield X, y, info_dict


class MNISTData(Benchmark):
    """
    Model evaluation using the benchmark vision dataset MNIST.

    Implements a data generation method returning a new evaluation data set for each scoring round.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    data_dir : str
        Directory containing calibration data obtained from MNIST classification.
    classifier_names : list
        Names of classifiers to be calibrated. Classification results on MNIST must be contained in `data_dir`.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    n_splits : int, default=10
        Number of splits for cross validation.
    test_size : float, default=0.9
        Size of test set.
    train_size : float, default=None
        Size of calibration set.
    random_state : int, RandomState instance or None, optional (default=None)
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, data_dir, classifier_names, cal_methods, cal_method_names,
                 n_splits=10, test_size=0.9, train_size=None, random_state=None):
        # Create cross validator which splits the data randomly based on test or train size
        cv = ms.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                             random_state=sklearn.utils.check_random_state(random_state))
        self.data_dir = data_dir
        self.classifier_names = classifier_names

        # Run super class constructor
        super().__init__(run_dir=run_dir, cal_methods=cal_methods, cal_method_names=cal_method_names, cross_validator=cv,
                         random_state=random_state)

    @staticmethod
    def classify_val_data(file, clf_name, classifier, output_folder='clf_output'):
        """
        Classify the MNIST evaluation data set with a given model.

        Parameters
        ----------
        file : str
            Output from the model is saved in the given directory.
        clf_name :
            Name of classifier.
        classifier : sklearn.base.BaseEstimator
            Classifier to classify MNIST data with.
        output_folder : str, default='clf_output'
            Name of folder where output is to be stored. This folder is created if non-existent.

        Returns
        -------

        """
        # Download data from https://www.openml.org/d/554
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)
        X = X / 255.  # normalize
        y = np.array(y, dtype=int)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=10000,
                                                                                    random_state=42)
        # Train classifier
        classifier.fit(X=X_train, y=y_train)

        # Classify validation data
        p_pred = classifier.predict_proba(X_test)
        y_y_pred = np.column_stack([y_test, np.argmax(p_pred, axis=1)])

        # Save to file
        os.makedirs(file, exist_ok=True)
        out_path = os.path.join(file, output_folder)
        os.makedirs(out_path, exist_ok=True)
        np.savetxt(os.path.join(out_path, "p_pred_{}.csv".format(clf_name)), p_pred, delimiter=",")
        np.savetxt(os.path.join(out_path, "y_y_pred_{}.csv".format(clf_name)), y_y_pred, fmt='%i', delimiter=",")

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Load classification results from file
        for clf_name in self.classifier_names:
            X = np.loadtxt(os.path.join(self.data_dir, 'p_pred_' + clf_name + '.csv'), dtype=float, delimiter=',')
            y = np.loadtxt(os.path.join(self.data_dir, 'y_y_pred_' + clf_name + '.csv'),
                           dtype=int, delimiter=',')[:, 0]

            # Calibration and test size
            if self.test_size > 1:
                calib_size_tmp = self.train_size
                test_size_tmp = self.test_size
            else:
                calib_size_tmp = int(np.shape(X)[0] * self.train_size)
                test_size_tmp = np.shape(X)[0] - int(np.shape(X)[0] * self.train_size)

            # Collect information on data set
            info_dict = {
                "Dataset": "MNIST",
                "Model": clf_name,
                "n_classes": np.shape(X)[1],
                "size": np.shape(X)[0],
                "calib_size": calib_size_tmp,
                "test_size": test_size_tmp}

            # Return (as generator)
            yield X, y, info_dict


class ImageNetData(Benchmark):
    """
    Model evaluation using the benchmark vision dataset ImageNet.

    Implements a data generation method returning a new evaluation data set for each scoring round.

    Parameters
    ----------
    run_dir : str
        Directory to run benchmarking in and save output and logs to.
    data_dir : str
        Directory containing calibration data obtained from ImageNet classification.
    classifier_names : list
        Names of classifiers to be calibrated. Classification results on ImageNet must be contained in `data_dir`.
    cal_methods : list
        Calibration methods to benchmark.
    cal_method_names : list
        Names of calibration methods.
    use_logits : bool, default=False
        Should the calibration methods be used with classification probabilities or logits.
    n_splits : int, default=10
        Number of splits for cross validation.
    test_size : float, default=0.9
        Size of test set.
    train_size : float, default=None
        Size of calibration set.
    random_state : int, RandomState instance or None, optional (default=None)
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, run_dir, data_dir, classifier_names, cal_methods, cal_method_names, use_logits=False,
                 n_splits=10, test_size=0.9, train_size=None, random_state=None):
        # Create cross validator which splits the data randomly based on test or train size
        cv = ms.ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                             random_state=sklearn.utils.check_random_state(random_state))
        self.data_dir = data_dir
        self.classifier_names = classifier_names
        self.use_logits = use_logits

        # Run super class constructor
        super().__init__(run_dir=run_dir, cal_methods=cal_methods, cal_method_names=cal_method_names, cross_validator=cv,
                         random_state=random_state)

    @staticmethod
    def classify_val_data(file, clf_name, n_classes=1000, file_ending="JPEG",
                          validation_folder='val', output_folder='clf_output'):
        """
        Classify the ImageNet evaluation data set with a given model.

        Parameters
        ----------
        file : str
            Directory containing ImageNet evaluation data. Folder 'val' under `file` is searched for images.
            Output from the model is saved in the given directory.
        clf_name : str
            Name of classifier (CNN architecture) to classify data with. Must be one of `pretrainedmodels.model_names`.
        n_classes : int, default=1000
            Number of classes of pretrained model on ImageNet.
        file_ending : str, default="JPEG"
            File suffix of images to classify.
        validation_folder : str, default='val'
            Folder where validation images are contained. Images must be contained in folder named after their class.
        output_folder : str, default='clf_output'
            Folder where output is stored.

        Returns
        -------

        """
        # Load pretrained model
        model = pretrainedmodels.__dict__[clf_name](num_classes=n_classes, pretrained='imagenet')

        # Data loading
        valdir = os.path.join(file, validation_folder)
        load_img = pretrainedmodels.utils.LoadImage()
        filenames = glob.glob(os.path.join(valdir, "**/*." + file_ending), recursive=True)

        # Image transformation
        tf_img = pretrainedmodels.utils.TransformImage(model)

        # Classify images
        n_images = len(filenames)
        all_logits = np.zeros([n_images, n_classes])
        all_y_y_pred = np.zeros([n_images, 2], dtype=int)

        with torch.no_grad():
            # Switch to evaluation mode
            model.eval()

            start_time = time.time()
            for i, path_img in enumerate(filenames):
                # Compute model output
                input_img = load_img(path_img)
                input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
                input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
                X = torch.autograd.Variable(input_tensor, requires_grad=False)
                logits = model(X)

                # Find class based on folder structure
                y = int(re.search(os.path.join(valdir, '(.+?)/.*'), path_img).group(1))

                # Collect model output
                all_logits[i, :] = logits.data.numpy()
                all_y_y_pred[i, :] = np.column_stack([y, np.argmax(logits.data.numpy(), axis=1)])

                # Print information on progress
                m, s = divmod(time.time() - start_time, 60)
                h, m = divmod(m, 60)
                print("Classifying image: {}/{:.0f}. {:.2f}% complete. Time elapsed: {:.0f}:{:02.0f}:{:02.0f}".format(
                    i + 1,
                    n_images,
                    (i + 1) / n_images * 100,
                    h, m, s)
                )

            # Save to file
            out_path = os.path.join(file, output_folder)
            os.makedirs(out_path, exist_ok=True)
            np.savetxt(os.path.join(out_path, "logits_{}.csv".format(clf_name)), all_logits, delimiter=",")
            np.savetxt(os.path.join(out_path, "y_y_pred_{}.csv".format(clf_name)), all_y_y_pred.astype(int), fmt='%i',
                       delimiter=",")

    def data_gen(self):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Load logits and classification results from file
        for clf_name in self.classifier_names:
            logits = np.loadtxt(os.path.join(self.data_dir, 'logits_' + clf_name + '.csv'), dtype=float, delimiter=',')
            y = np.loadtxt(os.path.join(self.data_dir, 'y_y_pred_' + clf_name + '.csv'),
                           dtype=int, delimiter=',')[:, 0]

            # Compute probabilities
            if not self.use_logits:
                X = scipy.special.softmax(logits, axis=1)
            else:
                X = logits

            # Calibration and test size
            if self.test_size > 1:
                calib_size_tmp = self.train_size
                test_size_tmp = self.test_size
            else:
                calib_size_tmp = int(np.shape(X)[0] * self.train_size)
                test_size_tmp = np.shape(X)[0] - int(np.shape(X)[0] * self.train_size)

            # Collect information on data set
            info_dict = {
                "Dataset": "ImageNet",
                "Model": clf_name,
                "n_classes": np.shape(X)[1],
                "size": np.shape(X)[0],
                "calib_size": calib_size_tmp,
                "test_size": test_size_tmp,
                "logits": self.use_logits
            }

            # Return (as generator)
            yield X, y, info_dict
