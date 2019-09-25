if __name__ == "__main__":
    import numpy as np
    import os.path
    import pycalib
    import pycalib.scoring
    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm
    import pycalib.texfig as texfig

    # ImageNet data
    random_state = 1
    n_classes = 1000
    file = "/home/j/Documents/research/projects/nonparametric_calibration/pycalib/data/imagenet/"
    output_folder = "clf_output"
    data_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")

    clf_name_dict = {
        'alexnet': "AlexNet",
        'vgg19': "VGG19",
        'resnet50': "ResNet50",
        'resnet152': "ResNet152",
        'densenet121': "DenseNet121",
        'densenet201': "DenseNet201",
        'inceptionv4': "InceptionV4",
        'se_resnext50_32x4d': "SE-ResNeXt50",
        'se_resnext101_32x4d': "SE-ResNeXt101",
        'polynet': "PolyNet",
        'senet154': "SENet154",
        'pnasnet5large': "PNASNET-5-Large",
        'nasnetalarge': "NASNet-A-Large"
    }

    clf_combinations = {
        "latent_maps_main": ["resnet152", "polynet", "pnasnet5large"],
        "latent_maps_appendix1": ["vgg19", "densenet201", 'nasnetalarge'],
        "latent_maps_appendix2": ['se_resnext50_32x4d', 'se_resnext101_32x4d', "senet154" ]
    }

    use_logits = True

    for clf_comb_name, clf_names in clf_combinations.items():

        # Benchmark data set
        imnet_benchmark = bm.ImageNetData(run_dir=run_dir, data_dir=data_dir,
                                          classifier_names=clf_names,
                                          cal_methods=[],
                                          cal_method_names=[],
                                          use_logits=use_logits, n_splits=10, test_size=10000,
                                          train_size=1000, random_state=random_state)
        # Filepath and plot axes
        folder_path = "/home/j/Documents/research/projects/nonparametric_calibration/" + \
                      "pycalib/figures/latent_functions/plots/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = folder_path + clf_comb_name + "_logits"

        fig, axes = texfig.subplots(width=7, ratio=.25, nrows=1, ncols=len(clf_names), w_pad=1)

        # Iterate through data sets, calibrate and plot latent functions
        for (i_clf, (Z, y, info_dict)) in enumerate(imnet_benchmark.data_gen()):

            # Train, test split
            cal_ind, test_ind = next(imnet_benchmark.cross_validator.split(Z, y))
            Z_cal = Z[cal_ind, :]
            y_cal = y[cal_ind]
            Z_test = Z[test_ind, :]
            y_test = y[test_ind]

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

            # Get latent function values
            z = np.linspace(start=3 * 10 ** -2, stop=1, num=1000)
            if use_logits:
                z = np.linspace(start=np.min(Z), stop=np.max(Z), num=1000)
            ts_latent = ts.latent(z)
            gpc_latent, gpc_latent_var = gpc.latent(z)

            if use_logits:
                xlims = np.array([np.min(z), np.max(z)])
                ylims = xlims

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
                axes[i_clf].set_title(clf_name_dict[clf_names[i_clf]])

            else:
                # Plot latent function corresponding to no calibration
                axes[i_clf].plot(z, np.log(z), '-', color='tab:gray', zorder=0,
                                 # label="Uncal: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_nocal))
                                 label="Uncal.")

                # Plot GPcalib latent function
                axes[i_clf].plot(z, gpc_latent,
                                 # label="GPcalib: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_gpc),
                                 label="GPcalib.",
                                 color='tab:blue',
                                 zorder=3)
                axes[i_clf].fill_between(z, gpc_latent - 2 * np.sqrt(gpc_latent_var),
                                         gpc_latent + 2 * np.sqrt(gpc_latent_var),
                                         color='tab:blue', alpha=.2, zorder=1)

                # Plot annotation
                handles, labels = axes[i_clf].get_legend_handles_labels()
                axes[i_clf].set_xlabel("probability score")

            # Plot annotation and legend
            if i_clf == 0:
                axes[i_clf].set_ylabel("latent function")
            if i_clf + 1 == len(clf_names):
                axes[i_clf].legend(loc="lower right", prop={'size': 9}, labelspacing=0.2)

        # Save plot to file
        texfig.savefig(filename)

    # TODO: plot latent functions for probability scores (MNIST) by removing use_logits=False code from above
