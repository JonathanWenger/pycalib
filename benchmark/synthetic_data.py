import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import pycalib.calibration_methods as cm
import pycalib.benchmark as bm
import pycalib.scoring as meas
from pycalib.plotting import reliability_diagram
import pycalib.texfig as texfig

if __name__ == "__main__":

    ###############################
    #   Benchmark
    ###############################

    ### Setup

    # Seed
    seed = 1

    # Output folder
    dir_out = "/home/j/Documents/research/projects/nonparametric_calibration/code/pycalib/data/synthetic_data/calibration"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    ### Generate calibration data

    # Size of data
    size_list = [1000, 10000]  # calibration data set size

    # Marginal probabilities
    n_classes = 5
    marginal_probs = np.ones(n_classes) / n_classes

    # Beta distribution parameters

    # random forests: probabilities concentrated around 1/n_classes
    params_forests = [2, 5]
    # neural networks: probabilities concentrated around 1
    params_NNs = [3, .75]
    beta_params = [params_forests, params_NNs]


    # Miscalibration functions
    def f_id(x):
        return x


    def f_root(x):
        return x ** (1 / 2)


    def f_power(x):
        return x ** 2


    miscal_function_names = {
        'identity': f_id,
        'root': f_root,
        "power": f_power
    }

    # Define calibration methods
    cal_methods_dict = {
        "No_Calibration": cm.NoCalibration(),
        "Platt_scaling": cm.PlattScaling(),
        "Isotonic_Regression": cm.IsotonicRegression(),
        "Beta_Calibration": cm.BetaCalibration(params='abm'),
        "Histogram_Binning": cm.HistogramBinning(mode='equal_freq'),
        "Bayesian_Binning_into_Quantiles": cm.BayesianBinningQuantiles(),
        "Temperature_Scaling": cm.TemperatureScaling(verbose=False),
        "GP_calibration": cm.GPCalibration(n_classes=n_classes, maxiter=300, n_inducing_points=100)
    }

    # Evaluate calibration methods
    sb = bm.SyntheticBeta(run_dir=dir_out,
                          cal_methods=list(cal_methods_dict.values()),
                          cal_method_names=list(cal_methods_dict.keys()),
                          beta_params=beta_params,
                          miscal_functions=list(miscal_function_names.values()),
                          miscal_function_names=list(miscal_function_names.keys()),
                          size=size_list,
                          marginal_probs=marginal_probs,
                          n_splits=10, test_size=0.9, random_state=seed)

    sb.run(n_jobs=1)

    #######################################
    ###        Plotting                 ###
    #######################################

    # Setup
    statistic = 'test_ECE'
    miscalibration_func = 'sum_sigmoids'
    cal_size = 1000
    cal_methods = [
        "NoCalibration",
        # "Platt_scaling",
        "Beta_Calibration",
        "Isotonic_Regression",
        # "Histogram_Binning",
        "Bayesian_Binning_into_Quantiles",
        "Temperature_Scaling",
        "GP_Calibration_affine"
    ]

    # Output folder
    dir_out = 'pycalib/out/synthetic_data/'
    if not os.path.exists(dir_out + '/plots'):
        os.makedirs(dir_out + '/plots')

    # Read CV calibration data from file
    cal_metrics_df = pd.read_csv(dir_out + '/cv_scores_gen.csv')


    def plot_calibration_comparison(cal_metrics_df, cal_methods, alpha, beta, beta_name, statistic, miscal_func,
                                    miscal_func_name,
                                    cal_size, dir_out):

        # --Plot confidence distribution-- #
        xx = np.linspace(0.5, 1, 1000)
        fig, ax = texfig.subplots()
        for bp in np.column_stack([alpha, beta]):
            ax.plot(xx, scipy.stats.beta.pdf(2 * xx - 1, a=bp[0], b=bp[1]),
                    label="alpha={:.1f}, beta={:.1f}".format(bp[0], bp[1]),
                    color="blue")
        ax.set_ylim([0, 5])
        ax.set_xlabel("confidence")
        ax.set_ylabel("density")
        ax.set_title("confidence histogram")

        texfig.savefig(filename=dir_out + '/plots/' + 'conf_dist_' + beta_name)

        # --Plot miscalibration function -- #
        fig, ax = texfig.subplots(ratio=1)
        ax.plot(xx, xx, linestyle="dashed", color="gray")
        ax.plot(xx, miscal_func(xx), label=miscal_func_name, color="red")
        ax.legend()
        ax.axis('equal')
        ax.set_xlabel('confidence')
        ax.set_ylabel('accuracy')
        ax.set_title("reliability diagram")

        texfig.savefig(filename=dir_out + '/plots/' + 'miscal_' + miscal_func_name)

        # --Plot given statistic -- #
        if len(np.unique(alpha)) <= len(np.unique(beta)):
            slice_param = np.unique(alpha)
            slice_param_name = 'alpha'
            x_param = beta
            x_param_name = 'beta'
        else:
            slice_param = np.unique(beta)
            slice_param_name = 'beta'
            x_param = alpha
            x_param_name = 'alpha'

        fig, ax = texfig.subplots(width=6, ratio=0.4, sharey='row')

        for i, sl_param in enumerate(slice_param):
            for calm in cal_methods:
                df_tmp = cal_metrics_df.loc[
                    (cal_metrics_df['model'] == calm) &
                    (cal_metrics_df['miscalibration_func'] == miscal_func_name) &
                    (cal_metrics_df['size'] == cal_size) &
                    (cal_metrics_df[slice_param_name] == sl_param) &
                    ([el in x_param for el in cal_metrics_df[x_param_name]])
                    ]
                ax.plot(df_tmp[x_param_name], np.abs(df_tmp[statistic + '.mean']), label=calm)
                ax.fill_between(df_tmp[x_param_name],
                                np.abs(df_tmp[statistic + '.mean']) + 2 * df_tmp[statistic + '.std'],
                                np.abs(df_tmp[statistic + '.mean']) - 2 * df_tmp[statistic + '.std'],
                                alpha=0.15)
                ax.set_title('{}={:.2f}'.format(slice_param_name, sl_param))
                ax.set_xlabel(x_param_name)
                # ax[i].set_yscale("log", nonposy='clip')

        ax.set_ylabel(statistic)
        ax.set_ylim(bottom=0, auto=True)
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

        # Save to file
        if not os.path.exists(dir_out + '/plots/' + statistic):
            os.makedirs(dir_out + '/plots/' + statistic)
        texfig.savefig(
            dir_out + '/plots/' + statistic + '/' + beta_name + '_' + miscal_func_name + '_' + statistic + '_' + str(
                cal_size))
        plt.close('all')


    # Save all combinations to file
    for stat in ['test_ECE', 'test_MCE', 'test_accuracy', 'test_overconfidence', 'test_underconfidence',
                 'test_log-loss', 'test_sharpness']:
        for miscal_f_name, miscal_f in miscal_function_names.items():
            for cal_size in np.unique(cal_metrics_df['size']):
                for beta_name, params in {"high_conf": params_NNs, "low_conf": params_forests}.items():
                    # TODO: make sure both beta distributions are plotted and not overwritten
                    plot_calibration_comparison(cal_metrics_df, cal_methods,
                                                alpha=params[:, 0], beta=params[:, 1],
                                                beta_name=beta_name,
                                                statistic=stat,
                                                miscal_func=miscal_f,
                                                miscal_func_name=miscal_f_name,
                                                cal_size=cal_size, dir_out=dir_out)


    def plot_feature_space_level_set(seed, dir_out='pycalib/out/synthetic_data/'):
        import sklearn.datasets
        from sklearn.neural_network import MLPClassifier
        import matplotlib.colors

        # Setup
        train_size = 1000
        cal_size = 100
        noise = .25
        contour_levels = 10

        # generate 2d classification dataset
        np.random.seed(seed)
        X, y = sklearn.datasets.make_circles(n_samples=train_size, noise=noise)

        # train classifier
        clf = MLPClassifier(hidden_layer_sizes=[10, 10], alpha=1, max_iter=200)
        clf.fit(X, y)

        # scatter plot, dots colored by class value
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        markers = {0: 'x', 1: '.'}

        fig, ax = texfig.subplots(width=8, ratio=.3, nrows=1, ncols=3, sharex=True, sharey=True)
        # grouped = df.groupby('label')
        # for key, group in grouped:
        #     group.plot(ax=ax[0], kind='scatter', x='x', y='y', label=key, marker=markers[key], color='gray', alpha=.75)

        # Put the result into a color plot
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            p_pred = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        else:
            p_pred = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z = p_pred[:, 1]
        Z0 = Z.reshape(xx.shape)
        cm = plt.cm.RdBu_r  # colormap
        cm_bright = matplotlib.colors.ListedColormap(['#FF0000', '#0000FF'])
        cont0 = ax[0].contourf(xx, yy, Z0, cmap=cm, alpha=.8, levels=contour_levels, vmin=0, vmax=1)
        ax[0].set_title("Classification Uncertainty")

        # calibrate
        X_cal, y_cal = sklearn.datasets.make_circles(n_samples=cal_size, noise=noise)
        p_cal = clf.predict_proba(X_cal)
        clf_cal = cm.GPCalibration(SVGP=True)
        clf_cal.fit(p_cal, y_cal)

        # calibrated contour plot
        Z1 = clf_cal.predict_proba(p_pred)[:, 1].reshape(xx.shape)
        cont1 = ax[1].contourf(xx, yy, Z1, cmap=cm, alpha=.8, levels=contour_levels, vmin=0, vmax=1)
        ax[1].set_title("Calibrated Uncertainty")

        # difference plot
        cm_diff = plt.cm.viridis_r  # colormap
        cont1 = ax[2].contourf(xx, yy, Z1 - Z0, cmap=cm_diff, alpha=.8)
        ax[2].set_title("Uncertainty Difference")

        # color bar
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([.96, 0.15, 0.05, 0.7])
        # cbar = fig.colorbar(cont1, cax=cbar_ax)

        # # contour labels
        # ax[0].clabel(cont0, inline=1, fontsize=8)
        # ax[1].clabel(cont1, inline=1, fontsize=8)

        texfig.savefig(dir_out + '/plots/' + 'level_sets')
