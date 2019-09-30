if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os.path
    import time
    import gpflow
    import pycalib
    import pycalib.scoring
    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm
    import pycalib.texfig as texfig

    # Initialization
    results_df = pd.DataFrame(columns=["data", "classifier", "cal_method", "time_inf", "time_pred"])
    results_list = list()

    # ImageNet data
    random_state = 1
    use_logits = True
    n_classes = 1000
    file = "/home/j/Documents/research/projects/nonparametric_calibration/code/pycalib/datasets/imagenet/"
    output_folder = "clf_output"
    data_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")

    clf_names = [
        # 'alexnet', 'vgg19',
        # 'resnet50', 'resnet152',
        # 'densenet121', 'densenet201',
        'inceptionv4',
        'se_resnext50_32x4d',
        'se_resnext101_32x4d'
    ]

    imagenet_benchmark_data = bm.ImageNetData(run_dir=run_dir, data_dir=data_dir,
                                              classifier_names=clf_names,
                                              cal_methods=[],
                                              cal_method_names=[],
                                              use_logits=use_logits, n_splits=1, test_size=10000,
                                              train_size=1000, random_state=random_state)
    with gpflow.defer_build():
        meanfunc = pycalib.gp_classes.ScalarMult()
        meanfunc.alpha.transform = gpflow.transforms.positive

    cal_methods = {
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=300, n_inducing_points=10,
                                      mean_function=meanfunc, logits=use_logits, verbose=False,
                                      random_state=random_state),
        "Temp": calm.TemperatureScaling()
    }

    # Iterate through data sets, calibrate and plot latent functions
    for Z, y, info_dict in imagenet_benchmark_data.data_gen():

        # Train, test split
        for i, (cal_index, test_index) in enumerate(imagenet_benchmark_data.cross_validator.split(Z)):
            print("CV iteration: {}".format(i + 1))
            Z_cal = Z[cal_index, :]
            y_cal = y[cal_index]
            Z_test = Z[test_index, :]
            y_test = y[test_index]

            for calib_name, calib_method in cal_methods.items():
                print("Timing " + calib_name)

                # Time inference
                inf_start = time.time()
                calib_method.fit(Z_cal, y_cal)
                inf_time = time.time() - inf_start

                # Time prediction
                pred_start = time.time()
                calib_method.predict_proba(Z_test)
                pred_time = time.time() - pred_start

                # Save to data frame
                results_list.append({"data": info_dict["Dataset"],
                                     "classifier": info_dict["Model"],
                                     "cal_method": calib_name,
                                     "time_inf": inf_time,
                                     "time_pred": pred_time})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(
        "/home/j/Documents/research/projects/nonparametric_calibration/code/" +
        "pycalib/figures/runtime_comparison/runtime_results_imagenet.csv")

    ################ MNIST

    classifier_names = [
        # "AdaBoost",
        # "XGBoost",
        # "mondrian_forest",
        "random_forest",
        "1layer_NN"
    ]
    file = "/home/j/Documents/research/projects/nonparametric_calibration/code/pycalib/datasets/mnist/"
    output_folder = "clf_output"
    data_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")
    mnist_benchmark_data = pycalib.benchmark.MNISTData(run_dir=run_dir, data_dir=data_dir,
                                                       classifier_names=classifier_names,
                                                       cal_methods=list([]),
                                                       cal_method_names=list([]),
                                                       n_splits=10, test_size=9000,
                                                       train_size=1000, random_state=random_state)

    cal_methods = {
        "GPcalib": calm.GPCalibration(n_classes=10, maxiter=300, logits=False, verbose=False,
                                      n_inducing_points=10, random_state=random_state),
        "Platt": calm.PlattScaling(random_state=random_state),
        "Isotonic": calm.IsotonicRegression(),
        "Beta": calm.BetaCalibration(),
        "BBQ": calm.BayesianBinningQuantiles(),
        "Temp": calm.TemperatureScaling()
    }

    # Iterate through data sets, calibrate and plot latent functions
    for Z, y, info_dict in mnist_benchmark_data.data_gen():

        # Train, test split
        for i, (cal_index, test_index) in enumerate(mnist_benchmark_data.cross_validator.split(Z)):
            print("CV iteration: {}".format(i + 1))
            Z_cal = Z[cal_index, :]
            y_cal = y[cal_index]
            Z_test = Z[test_index, :]
            y_test = y[test_index]

            for calib_name, calib_method in cal_methods.items():
                print("Timing " + calib_name)

                # Time inference
                inf_start = time.time()
                calib_method.fit(Z_cal, y_cal)
                inf_time = time.time() - inf_start

                # Time prediction
                pred_start = time.time()
                calib_method.predict_proba(Z_test)
                pred_time = time.time() - pred_start

                # Save to data frame
                results_list.append({"data": info_dict["Dataset"],
                                     "classifier": info_dict["Model"],
                                     "cal_method": calib_name,
                                     "time_inf": inf_time,
                                     "time_pred": pred_time})

    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(
        "/home/j/Documents/research/projects/nonparametric_calibration/code/" +
        "pycalib/figures/runtime_comparison/runtime_results_MNIST.csv")

    ###### Plot

    # Load data
    results_df = pd.read_csv("/home/j/Documents/research/projects/nonparametric_calibration/code/" +
                             "pycalib/figures/runtime_comparison/runtime_results_MNIST.csv", index_col=0)

    plot_classifier = "1layer_NN"
    plot_df = results_df.loc[results_df['classifier'] == plot_classifier]

    grouped = plot_df.groupby(["cal_method", "classifier", "data"])
    summ_stats = grouped.agg([np.mean, np.std])

    cal_methods = list(summ_stats.index.levels[0])
    mean_time_inf = list(summ_stats["time_inf"]["mean"])
    mean_time_pred = list(summ_stats["time_pred"]["mean"])

    # Plot
    fig, axs = texfig.subplots(nrows=1, ncols=2, width=5, ratio=.5, sharey=True)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].scatter(cal_methods, mean_time_inf)
    axs[1].scatter(cal_methods, mean_time_pred)
    minaxis = np.min(np.concatenate([mean_time_inf, mean_time_pred]))
    maxaxis = np.max(np.concatenate([mean_time_inf, mean_time_pred]))
    axs[0].set_ylim([minaxis - .0001, maxaxis + 1])

    axs[0].title.set_text('Inference')
    axs[1].title.set_text('Prediction')
    axs[0].set_ylabel("seconds")

    # Rotate tick labels
    for ax in axs:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    # Save plot to file
    texfig.savefig("/home/j/Documents/research/projects/nonparametric_calibration/code/" +
                   "pycalib/figures/runtime_comparison/runtime_results_MNIST")
