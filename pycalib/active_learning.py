import numpy as np
import pandas as pd
import os
import time
import gpflow
import scipy.stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sklearn.utils
import pycalib.texfig as texfig
import pycalib.scoring as sc


def query_uncertainty(p_pred):
    """
    Calculates the uncertainty of the prediction probabilities.

    Parameters
    ----------
    p_pred :
        Predicted probabilities.

    Returns
    -------
    Uncertainty of the predicted probabilities.

    """
    return 1 - np.max(p_pred, axis=1)


def query_margin(p_pred):
    """
    Calculates the margin of the prediction probabilities.

    Parameters
    ----------
    p_pred :
        Predicted probabilities.

    Returns
    -------
    Margin of the predicted probabilities.

    """
    if p_pred.shape[1] == 1:
        return np.zeros(shape=len(p_pred))

    p_pred_partition = np.partition(-p_pred, 1, axis=1)
    margin = - p_pred_partition[:, 0] + p_pred_partition[:, 1]

    return margin


def query_norm_entropy(p_pred):
    """
    Calculates the normalized entropy of the prediction probabilities. The output is normalized to [0,1].

    Parameters
    ----------
    p_pred :
        Predicted probabilities.

    Returns
    -------
    Normalized entropy of the predicted probabilities.

    """
    return np.transpose(entropy(np.transpose(p_pred))) / np.log(p_pred.shape[1])


class ActiveLearningExperiment(object):
    """
    Experiment testing the effects of calibration on active learning
    """

    def __init__(self, classifier, X_train, y_train, X_test, y_test,
                 query_criterion, uncertainty_thresh, pretrain_size,
                 calibration_method, calib_size, calib_points, batch_size):
        """
        Experiment testing the effects of calibration on active learning

        Parameters
        ----------
        classifier
        X_train
        y_train
        X_test
        y_test
        query_criterion
        uncertainty_thresh
        pretrain_size
        calibration_method
        calib_size
        calib_points
        batch_size
        """
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes = np.sort(np.unique(np.hstack([y_train, y_test])))
        self.query_criterion = query_criterion
        self.uncertainty_thresh = uncertainty_thresh
        self.pretrain_size = pretrain_size
        self.calibration_method = calibration_method
        assert calib_points[0] >= pretrain_size, "Calibration cannot happen before pre-training."
        self.calib_size = calib_size
        self.calib_points = np.array(calib_points)
        assert batch_size < np.shape(X_train)[0], "Batch size cannot be larger than the training set."
        self.batch_size = batch_size
        self.metrics = {
            "error": (sc.error, {}),
            # "precision": (sc.precision, {"labels": self.classes, "average": "weighted"}),
            # "recall": (sc.recall, {"labels": self.classes, "average": "weighted"}),
            "log-loss": (sklearn.metrics.log_loss, {"labels": self.classes}),
            "$\\text{ECE}_1$": (sc.expected_calibration_error, {"p": 1}),
            "MCE": (sc.expected_calibration_error, {"p": np.inf}),
            "sharpness": (sc.sharpness, {}),
            "overconfidence": (sc.overconfidence, {}),
            "underconfidence": (sc.underconfidence, {}),
            "$\\frac{o(f)}{u(f)}$": (sc.ratio_over_underconfidence, {}),
            "$\\frac{\\text{accuracy}}{\\text{error}}$": (sc.odds_correctness, {}),
            "$\\abs{o(f) \\mathbb{P}(\\hat{y} \\neq y) -u(f) \\mathbb{P}(\\hat{y} = y)}$": (
                sc.weighted_abs_conf_difference(), {}),
            "$avg. confidence": (sc.average_confidence, {})
        }
        self.result_df = pd.DataFrame()

    def _active_learning(self, X_train_shuffled, y_train_shuffled, classifier, calibration_method, pretrain_ind,
                         do_calibration=True):
        """
        Perform one pass of active learning through data set

        Parameters
        ----------
        X_train_shuffled
        y_train_shuffled
        classifier
        calibration_method
        pretrain_ind
        do_calibration

        Returns
        -------
        results : dict

        """

        # Initialization
        train_indices = list(pretrain_ind)
        calib_indices = []
        calib_set = None
        next_calib_point_ind = 0
        results = {
            "n_samples_queried": [],
            "n_samples": []
        }
        for metric in self.metrics.keys():
            results[metric] = []

        # Pre-train
        classifier.partial_fit(X_train_shuffled[pretrain_ind, :],
                               y_train_shuffled[pretrain_ind],
                               classes=self.classes)

        # Step through training set
        i = self.pretrain_size
        while i < np.shape(X_train_shuffled)[0]:

            # Predict on test set
            p_pred_test = classifier.predict_proba(self.X_test)
            if do_calibration:
                try:
                    p_pred_test = calibration_method.predict_proba(p_pred_test)
                except sklearn.exceptions.NotFittedError:
                    pass

            # Evaluate metrics
            for key in self.metrics.keys():
                # Evaluate metric and save
                metric, kwargs = self.metrics[key]
                results[key].append(metric(self.y_test, p_pred_test, **kwargs))
            results["n_samples"].append(i)
            results["n_samples_queried"].append(len(train_indices) + len(calib_indices))
            assert i >= len(train_indices) + len(
                calib_indices), "Seen data cannot be less than train and calibration data combined."

            # Predict on batch
            ind_end = np.minimum(i + self.batch_size, np.shape(X_train_shuffled)[0])
            p_pred_batch = classifier.predict_proba(X_train_shuffled[i:ind_end])
            if do_calibration:
                try:
                    p_pred_batch = calibration_method.predict_proba(p_pred_batch)
                except sklearn.exceptions.NotFittedError:
                    pass

            # Evaluate query criterion
            query_unc = self.query_criterion(p_pred_batch)
            # Find samples with higher query uncertainty than the threshold
            active_sample_ids = np.where(query_unc > self.uncertainty_thresh)[0] + i

            # Train on queried samples
            if do_calibration and next_calib_point_ind < len(self.calib_points):
                clip_bool = results["n_samples_queried"][-1] + np.arange(len(active_sample_ids)) + 1 < \
                            self.calib_points[next_calib_point_ind]
                clipped_active_sample_ids = active_sample_ids[clip_bool]
            else:
                clipped_active_sample_ids = active_sample_ids
            if len(clipped_active_sample_ids) > 0:
                train_indices.extend(clipped_active_sample_ids)
                classifier.partial_fit(X_train_shuffled[clipped_active_sample_ids, :],
                                       y_train_shuffled[clipped_active_sample_ids])

            # Calibrate if at calibration point
            if do_calibration and next_calib_point_ind < len(self.calib_points) and results["n_samples_queried"][
                -1] + len(active_sample_ids) >= self.calib_points[next_calib_point_ind]:
                # Query all remaining samples
                # TODO: this is inefficient as only a few of the remaining samples will need to be queried.
                p_pred_remain = classifier.predict_proba(X_train_shuffled[self.calib_points[next_calib_point_ind]:, :])
                try:
                    p_pred_remain = calibration_method.predict_proba(p_pred_remain)
                except sklearn.exceptions.NotFittedError:
                    pass
                query_unc = self.query_criterion(p_pred_remain)
                # Select samples with higher query uncertainty than the threshold
                calib_sample_ids = np.where(query_unc > self.uncertainty_thresh)[0][0:self.calib_size] + \
                                   self.calib_points[next_calib_point_ind]

                # Calibrate on calibration set
                calib_indices.extend(calib_sample_ids)
                if next_calib_point_ind == 0:
                    calib_set = classifier.predict_proba(X_train_shuffled[calib_sample_ids, :])
                else:
                    calib_set = np.concatenate(
                        (calib_set, classifier.predict_proba(X_train_shuffled[calib_sample_ids, :])),
                        axis=0)
                calibration_method.fit(calib_set, y_train_shuffled[calib_indices])

                # Adjust calibration point, samples queried and training index
                next_calib_point_ind += 1
                i = calib_sample_ids[self.calib_size - 1] + 1
            else:
                # Update training index
                i += self.batch_size

        return results

    def _training_shuffle(self, n_runs, random_state):
        """
        Returns the full dataset or a generator of datasets.

        Returns
        -------
            X, y giving uncalibrated predictions and corresponding classes.
        """
        # Seed
        random_state = sklearn.utils.check_random_state(random_state)

        for n in range(n_runs):
            # Shuffle training set
            trainset = np.column_stack([self.X_train, self.y_train])
            random_state.shuffle(trainset)
            X_train_shuffled = trainset[:, 0:-1]
            y_train_shuffled = trainset[:, -1].astype(int)

            # Return (as generator)
            yield X_train_shuffled, y_train_shuffled, n

    def run(self, n_cv=10, random_state=None):

        # Initialization
        random_state = sklearn.utils.check_random_state(random_state)
        print("Starting active learning experiment.")
        start_time = time.time()
        for X_train, y_train, n_run in self._training_shuffle(n_cv, random_state):
            cv_time = time.time()

            # Active learning
            results = self._active_learning(X_train_shuffled=X_train,
                                            y_train_shuffled=y_train,
                                            classifier=sklearn.clone(self.classifier),
                                            calibration_method=sklearn.clone(self.calibration_method),
                                            pretrain_ind=range(self.pretrain_size),
                                            do_calibration=False)

            # Active learning with calibration
            pretrain_ind = range(np.min(np.concatenate((np.array([self.pretrain_size]), np.array(self.calib_points)))))
            results_calib = self._active_learning(X_train_shuffled=X_train,
                                                  y_train_shuffled=y_train,
                                                  classifier=sklearn.clone(self.classifier),
                                                  calibration_method=sklearn.clone(self.calibration_method),
                                                  pretrain_ind=pretrain_ind)

            # Save to dataframe
            df_nocalib = pd.DataFrame(results)
            df_nocalib["calibration_method"] = ""
            df_calib = pd.DataFrame(results_calib)
            df_calib["calibration_method"] = self.calibration_method.__class__.__name__
            df_tmp = df_nocalib.append(df_calib)
            df_tmp["cv_run"] = n_run
            self.result_df = self.result_df.append(df_tmp)

            # Log summary statistics for fold
            m, s = divmod(time.time() - start_time, 60)
            h, m = divmod(m, 60)
            m_cv, s_cv = divmod(time.time() - cv_time, 60)
            h_cv, m_cv = divmod(m_cv, 60)
            print("CV run {}/{}\t Time: {:.0f}:{:02.0f}:{:02.0f}, Total:{:.0f}:{:02.0f}:{:02.0f}".format(
                n_run + 1,
                n_cv,
                h_cv, m_cv, s_cv,
                h, m, s
            ))

        return self.result_df

    def save_result(self, file):
        """
        Save experiment results to file.

        Parameters
        ----------
        file : str
            Filepath to save results at

        Returns
        -------

        """
        # Save data to file
        self.result_df.to_csv(os.path.join(file, "active_learning_results.csv"), sep='\t', index=False)

    @staticmethod
    def load_result(file):
        """
        Load results from file.

        Parameters
        ----------
        file

        Returns
        -------

        """
        # Read and remove Nan
        df = pd.read_csv(file, sep='\t', index_col=False)
        df = df.fillna("")
        return df

    def plot(self, file, metrics_list, width=None, height=None, scatter=False, confidence=True):
        """
        Plots the given list of metrics for the active learning experiment.

        Parameters
        ----------
        file : str
            Filename of resulting plot.
        metrics_list : list
            List of names of metrics to plot.
        Returns
        -------

        """
        # Initialization
        n_subplots = len(metrics_list)

        # Generate curves for the given metrics
        if width is None:
            width = 3.25 * n_subplots
        if height is None:
            height = 1.7
        fig, axes = texfig.subplots(ncols=n_subplots, width=width, ratio=height * 1.0 / width, w_pad=1)
        cmap = get_cmap("tab10")  # https://matplotlib.org/gallery/color/colormap_reference.html

        # Allow for one metric only
        if n_subplots == 1:
            axes = [axes]

        for ax_ind, metric_name in enumerate(metrics_list):

            # Plot calibration ranges
            for cp in self.calib_points:
                axes[ax_ind].axvspan(cp, cp + self.calib_size, alpha=0.3, color='gray', linewidth=0.0)

            # Plot metric curves
            for i_calmethod, calib_method in enumerate(np.unique(self.result_df["calibration_method"])):
                # Extract relevant data
                df_tmp = self.result_df.loc[self.result_df["calibration_method"] == calib_method]
                xdata = np.array(df_tmp["n_samples_queried"], dtype="float64")
                ydata = np.array(df_tmp[metric_name], dtype="float64")

                # Compute average samples queried and restrict plot
                grouped = df_tmp.groupby(["calibration_method", "cv_run"])
                summ_stats = grouped.agg(np.max)
                mean_n_samples_queried = np.mean(summ_stats["n_samples_queried"])
                std_n_samples_queried = np.std(summ_stats["n_samples_queried"])
                print("{} samples queried: {} +- {}".format(calib_method,
                                                            mean_n_samples_queried,
                                                            std_n_samples_queried))
                # Fit GP to result
                k = gpflow.kernels.Matern52(input_dim=1, lengthscales=1000, variance=.1) + gpflow.kernels.White(
                    input_dim=1,
                    variance=.01)
                m = gpflow.models.GPR(xdata.reshape(-1, 1), ydata.reshape(-1, 1), kern=k)
                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(m)

                # Predict GP mean up to mean of n_samples_qeried
                xx = np.linspace(np.min(xdata), mean_n_samples_queried, 1000).reshape(-1, 1)
                mean, var = m.predict_f(xx)
                axes[ax_ind].plot(xx, mean, color=cmap.colors[i_calmethod], zorder=10,
                                  label="MF entropy " + calib_method)
                if scatter:
                    axes[ax_ind].scatter(xdata, ydata, color=cmap.colors[i_calmethod], s=4, alpha=0.3,
                                         edgecolors='none')
                if confidence:
                    axes[ax_ind].fill_between(xx[:, 0],
                                              mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                                              mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), alpha=0.3, linewidth=0.0)

            # Set labels and legend
            axes[ax_ind].set_xlabel("queried samples")
            axes[ax_ind].set_ylabel(metric_name)

        axes[ax_ind].legend(prop={'size': 9}, labelspacing=0.2)

        # Save plot to file
        texfig.savefig(os.path.join(file))
        plt.close("all")
