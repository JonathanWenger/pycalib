if __name__ == "__main__":
    import numpy as np
    import os.path
    import gpflow
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

    clf_names = [
        # 'alexnet', 'vgg19',
        # 'resnet50', 'resnet152',
        'densenet121',
        'densenet201',
        'inceptionv4',
        'se_resnext50_32x4d',
        'se_resnext101_32x4d',
        'polynet',
        'pnasnet5large'
    ]

    for use_logits in [True, False]:

        imnet_benchmark = bm.ImageNetData(run_dir=run_dir, data_dir=data_dir,
                                          classifier_names=clf_names,
                                          cal_methods=[],
                                          cal_method_names=[],
                                          use_logits=use_logits, n_splits=10, test_size=10000,
                                          train_size=1000, random_state=random_state)

        # Iterate through data sets, calibrate and plot latent functions
        for Z, y, info_dict in imnet_benchmark.data_gen():

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

            # Compute calibration error
            ECE_nocal = pycalib.scoring.expected_calibration_error(y_test, nocal.predict_proba(Z_test))
            ECE_ts = pycalib.scoring.expected_calibration_error(y_test, ts.predict_proba(Z_test))
            ECE_gpc = pycalib.scoring.expected_calibration_error(y_test, gpc.predict_proba(Z_test))

            # Get latent function values
            z = np.linspace(start=3 * 10 ** -2, stop=1, num=1000)
            if use_logits:
                z = np.linspace(start=np.min(Z), stop=np.max(Z), num=1000)
            ts_latent = ts.latent(z)
            gpc_latent, gpc_latent_var = gpc.latent(z)

            # Filepath and plot axes
            logit_str = "_probs"
            if use_logits:
                logit_str = "_logits"
            folder_path = "/home/j/Documents/research/projects/nonparametric_calibration/" + \
                          "pycalib/figures/latent_functions/plots/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            filename = folder_path + info_dict['Dataset'] + "_" + info_dict['Model'] + logit_str

            fig, axes = texfig.subplots(nrows=1, ncols=1, width=3, ratio=.8, sharex=True)

            if use_logits:
                # Compute y-intercepts by minimizing L2 distance between latent functions and identity
                y_intercept_ts = 0.5 * (1 - ts.T) * (np.max(z) + np.min(z))
                y_intercept_gpcalib = np.mean(gpc_latent - z)

                # Plot shifted latent function: GPcalib
                axes.plot(z, gpc_latent + y_intercept_gpcalib, label="GPcalib: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_gpc),
                          color='tab:blue',
                          zorder=10)
                axes.fill_between(z, gpc_latent + y_intercept_gpcalib - 2 * np.sqrt(gpc_latent_var),
                                  gpc_latent + y_intercept_gpcalib + 2 * np.sqrt(gpc_latent_var),
                                  color='tab:blue', alpha=.2, zorder=9)

                # Plot shifted latent function: temperature scaling
                axes.plot(z, ts_latent + y_intercept_ts, '-', color='tab:orange',
                          label="Temp: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_ts),
                          zorder=3)

                xlims = np.array([np.min(z), np.max(z)])
                ylims = np.array([np.min(gpc_latent), np.max(gpc_latent)])

                # Plot latent function of no calibration
                axes.plot(np.clip(xlims, min(ylims), max(ylims)),
                          np.clip(xlims, min(ylims), max(ylims)), '-', color='tab:gray', zorder=3,
                          label="Uncal: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_nocal))

                # Plot annotation
                handles, labels = axes.get_legend_handles_labels()
                axes.set_xlabel("logit")
                axes.set_xlim(xlims[0], xlims[1])
                axes.set_ylim(ylims[0], ylims[1])
                handles = [handles[1], handles[0], handles[2]]
                labels = [labels[1], labels[0], labels[2]]

            else:
                # Plot GPcalib latent function
                axes.plot(z, gpc_latent, label="GPcalib: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_gpc),
                          color='tab:blue',
                          zorder=10)
                axes.fill_between(z, gpc_latent - 2 * np.sqrt(gpc_latent_var), gpc_latent + 2 * np.sqrt(gpc_latent_var),
                                  color='tab:blue', alpha=.2, zorder=9)

                # Plot latent function corresponding to no calibration
                axes.plot(z, np.log(z), '-', color='tab:gray', zorder=3,
                          label="Uncal: $\\textup{ECE}_1" + "={:.4f}$".format(ECE_nocal))

                # Plot annotation
                handles, labels = axes.get_legend_handles_labels()
                axes.set_xlabel("probability score")
                axes.set_xlim(0, 1)
                axes.set_ylim(np.log(np.min(z)), np.log(np.max(z)))
                handles = [handles[1], handles[0]]
                labels = [labels[1], labels[0]]

            # Plot annotation and legend
            axes.set_ylabel("latent function")
            axes.legend(handles, labels, loc="lower right", prop={'size': 9})

            # Save plot to file
            texfig.savefig(filename)
