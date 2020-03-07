"""Compare latent functions of temperature scaling and GP calibration visually on different benchmark datasets."""

if __name__ == "__main__":

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm
    import pycalib.plotting
    import pycalib.texfig as texfig

    # Setup
    plot_hist = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(dir_path)) == "pycalib":
        dir_path += "/datasets/"
    else:
        dir_path = os.path.split(os.path.split(dir_path)[0])[0] + "/datasets/"

    # Seed
    random_state = 1

    for dataset in ["ImageNet"]:  # "MNIST", "CIFAR100", "ImageNet"]:

        if dataset == "ImageNet":
            # Classifier display names
            clf_name_dict = {
                'alexnet': "AlexNet",
                'vgg19': "VGG19",
                'resnet50': "ResNet-50",
                'resnet152': "ResNet-152",
                'densenet121': "DenseNet-121",
                'densenet201': "DenseNet-201",
                'inceptionv4': "InceptionV4",
                'se_resnext50_32x4d': "SE-ResNeXt-50",
                'se_resnext101_32x4d': "SE-ResNeXt-101",
                'polynet': "PolyNet",
                'senet154': "SENet-154",
                'pnasnet5large': "PNASNet-5-Large",
                'nasnetalarge': "NASNet-A-Large"
            }

            # Classifier combinations to plot
            clf_combinations = {
                "latent_maps_main": ["resnet152", "polynet", "pnasnet5large"],
                "latent_maps_appendix1": ["vgg19", "densenet201", 'nasnetalarge'],
                "latent_maps_appendix2": ['se_resnext50_32x4d', 'se_resnext101_32x4d', "senet154"]
            }

            # Dataset setup
            n_classes = 1000
            file = dir_path + "/imagenet/"
            use_logits = True

        elif dataset == "MNIST":
            clf_name_dict = {
                "AdaBoost": "AdaBoost",
                "XGBoost": "XGBoost",
                "mondrian_forest": "Mondrian Forest",
                "random_forest": "Random Forest",
                "1layer_NN": "1-layer NN"
            }

            clf_combinations = {
                "latent_maps_appendix1": ["AdaBoost", "mondrian_forest", "1layer_NN"],
                "latent_maps_appendix2": ["XGBoost", "random_forest"]
            }

            # Dataset setup
            n_classes = 10
            file = dir_path + "/mnist/"
            use_logits = False

        elif dataset == "CIFAR100":
            clf_name_dict = {
                'alexnet': "AlexNet",
                'WRN-28-10-drop': "WideResNet",
                'resnext-8x64d': "ResNeXt-29 (8x64)",
                'resnext-16x64d': "ResNeXt-29 (16x64)",
                'densenet-bc-l190-k40': "DenseNet-BC-190"
            }

            clf_combinations = {
                "latent_maps_cifar1": ["alexnet", "WRN-28-10-drop", "densenet-bc-l190-k40"],
                "latent_maps_cifar2": ["resnext-8x64d", "resnext-16x64d"]
            }

            # Data set setup
            n_classes = 100
            file = dir_path + "/cifar100/"
            output_folder = "clf_output"
            use_logits = True
        else:
            raise NotImplementedError("Dataset chosen is not available.")

        # Filepaths
        output_folder = "clf_output"
        data_dir = os.path.join(file, output_folder)
        run_dir = os.path.join(file, "calibration")

        for clf_comb_name, clf_names in clf_combinations.items():

            # Benchmark data set
            if dataset == "ImageNet":
                benchmark = bm.ImageNetData(run_dir=run_dir, clf_output_dir=data_dir,
                                            classifier_names=clf_names,
                                            cal_methods=[],
                                            cal_method_names=[],
                                            use_logits=use_logits, n_splits=10, test_size=10000,
                                            train_size=1000, random_state=random_state)
            elif dataset == "MNIST":
                benchmark = pycalib.benchmark.MNISTData(run_dir=run_dir, clf_output_dir=data_dir,
                                                        classifier_names=clf_names,
                                                        cal_methods=[],
                                                        cal_method_names=[],
                                                        n_splits=10, test_size=9000,
                                                        train_size=1000, random_state=random_state)
            elif dataset == "CIFAR100":
                benchmark = pycalib.benchmark.CIFARData(run_dir=run_dir, clf_output_dir=data_dir,
                                                        classifier_names=clf_names,
                                                        cal_methods=[],
                                                        cal_method_names=[],
                                                        n_splits=10, test_size=9000,
                                                        use_logits=use_logits,
                                                        train_size=1000, random_state=random_state)
            else:
                raise NotImplementedError("Dataset choice not available.")

            # Filepath and plot axes
            folder_path = os.path.dirname(os.path.realpath(__file__))
            if os.path.basename(os.path.normpath(folder_path)) == "pycalib":
                folder_path += "/figures/latent_functions"
            folder_path += "/plots/latent_maps/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_suffix = "_probs"
            if use_logits:
                file_suffix = "_logits"
            filename = folder_path + clf_comb_name + file_suffix

            fig, axes = texfig.subplots(width=7 / 3 * len(clf_names), ratio=.25 * 3 / len(clf_names),
                                        nrows=1, ncols=len(clf_names), w_pad=1)

            # Iterate through data sets, calibrate and plot latent functions
            for (i_clf, (Z, y, info_dict)) in enumerate(benchmark.data_gen()):

                # Train, test split
                cal_ind, test_ind = next(benchmark.cross_validator.split(Z, y))
                Z_cal = Z[cal_ind, :]
                y_cal = y[cal_ind]
                Z_test = Z[test_ind, :]
                y_test = y[test_ind]
                hist_data = Z_cal.flatten()

                # Calibrate
                nocal = calm.NoCalibration(logits=use_logits)

                ts = calm.TemperatureScaling()
                ts.fit(Z_cal, y_cal)

                gpc = calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                         logits=use_logits, verbose=True,
                                         random_state=random_state)
                gpc.fit(Z_cal, y_cal)

                # # Compute calibration error
                # ECE_nocal = pycalib.scoring.expected_calibration_error(y_test, nocal.predict_proba(Z_test), n_bins=100)
                # ECE_ts = pycalib.scoring.expected_calibration_error(y_test, ts.predict_proba(Z_test), n_bins=100)
                # ECE_gpc = pycalib.scoring.expected_calibration_error(y_test, gpc.predict_proba(Z_test), n_bins=100)

                # Plot reliability diagrams
                if not os.path.exists(os.path.join(folder_path, "reliability_diagrams")):
                    os.makedirs(os.path.join(folder_path, "reliability_diagrams"))
                p_pred_nocal = nocal.predict_proba(Z_test)
                p_pred_ts = ts.predict_proba(Z_test)
                p_pred_gpc = gpc.predict_proba(Z_test)
                pycalib.plotting.reliability_diagram(y=y_test, p_pred=[p_pred_nocal, p_pred_ts, p_pred_gpc], n_bins=15,
                                                     show_ece=False, show_legend=False,
                                                     title=["Uncal.", "Temp.", "GPcalib"],
                                                     model_name=None, plot_width=2.2 * 3 + .2, plot_height=2.2,
                                                     filename=os.path.join(folder_path, "reliability_diagrams",
                                                                           info_dict['Model'] + "_reldiagram"))

                # Get latent function values
                z = np.linspace(start=3 * 10 ** -2, stop=1, num=1000)
                if use_logits:
                    z = np.linspace(start=np.min(Z), stop=np.max(Z), num=1000)
                ts_latent = ts.latent(z)
                gpc_latent, gpc_latent_var = gpc.latent(z)

                if use_logits:

                    # Compute factor for histogram height
                    xlims = np.array([np.min(z), np.max(z)])
                    ylims = xlims
                    factor = 1.5 * ylims[1] / len(hist_data)

                    # Plot latent function of no calibration
                    axes[i_clf].plot(xlims, ylims, '-', color='tab:gray', zorder=0,
                                     # label="Uncal: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_nocal))
                                     label="Uncal.")

                    # Compute y-intercepts by minimizing L2 distance between latent functions and identity
                    y_intercept_ts = 0.5 * (1 - ts.T) * (np.max(z) + np.min(z))
                    y_intercept_gpcalib = np.mean(gpc_latent - z)

                    # Plot shifted latent function: temperature scaling
                    ts_latent_shifted = ts_latent + y_intercept_ts
                    ts_indices = (ts_latent_shifted <= ylims[1]) & (
                            ts_latent_shifted >= ylims[0])  # Only display points in plot area
                    axes[i_clf].plot(z[ts_indices], ts_latent_shifted[ts_indices],
                                     '-', color='tab:orange',
                                     # label="Temp: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_ts),
                                     label="Temp.", zorder=2)

                    # Plot shifted latent function: GPcalib
                    gpc_latent_shifted = gpc_latent + y_intercept_gpcalib
                    gpc_indices = (gpc_latent_shifted <= ylims[1]) & (
                            gpc_latent_shifted >= ylims[0])  # Only display points in plot area
                    axes[i_clf].plot(z[gpc_indices], gpc_latent_shifted[gpc_indices],
                                     # label="GPcalib: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_gpc),
                                     label="GPcalib", color='tab:blue',
                                     zorder=3)
                    axes[i_clf].fill_between(z,
                                             np.clip(a=gpc_latent + y_intercept_gpcalib - 2 * np.sqrt(gpc_latent_var),
                                                     a_min=ylims[0], a_max=ylims[1]),
                                             np.clip(a=gpc_latent + y_intercept_gpcalib + 2 * np.sqrt(gpc_latent_var),
                                                     a_min=ylims[0], a_max=ylims[1]),
                                             color='tab:blue', alpha=.2, zorder=1)

                    # Plot annotation
                    axes[i_clf].set_xlabel("logit")

                else:
                    # Plot latent function corresponding to no calibration
                    axes[i_clf].plot(z, np.log(z), '-', color='tab:gray', zorder=1,
                                     # label="Uncal: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_nocal))
                                     label="Uncal.")

                    # Compute y-intercepts by minimizing L2 distance between latent functions and identity
                    y_intercept_ts = np.mean(np.log(z) - ts_latent)
                    # y_intercept_gpcalib = np.mean(gpc_latent - np.log(z))

                    # Plot shifted latent function: temperature scaling
                    ts_latent_shifted = ts_latent + y_intercept_ts
                    axes[i_clf].plot(z, ts_latent_shifted,
                                     '-', color='tab:orange',
                                     # label="Temp: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_ts),
                                     label="Temp.", zorder=3)

                    # Plot GPcalib latent function
                    axes[i_clf].plot(z, gpc_latent,
                                     # label="GPcalib: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_gpc),
                                     label="GPcalib.",
                                     color='tab:blue',
                                     zorder=3)
                    axes[i_clf].fill_between(z, gpc_latent - 2 * np.sqrt(gpc_latent_var),
                                             gpc_latent + 2 * np.sqrt(gpc_latent_var),
                                             color='tab:blue', alpha=.2, zorder=2)

                    # Compute factor for histogram height
                    ylims = axes[i_clf].get_ylim()
                    factor = 1.5 * ylims[1] / len(hist_data)

                    # Plot annotation
                    axes[i_clf].set_xlabel("probability score")

                # Plot histogram
                if plot_hist:
                    axes[i_clf].hist(hist_data, weights=factor * np.ones_like(hist_data),
                                     bottom=ylims[0], zorder=0, color="gray", alpha=0.4)

                # Plot annotation and legend
                axes[i_clf].set_title(clf_name_dict[clf_names[i_clf]])
                if i_clf == 0:
                    axes[i_clf].set_ylabel("latent function")
                if i_clf + 1 == len(clf_names):
                    axes[i_clf].legend(prop={'size': 9}, labelspacing=0.2)  # loc="lower right"

            # Save plot to file
            texfig.savefig(filename)
            plt.close("all")
