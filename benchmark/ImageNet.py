"""Calibration experiment on the Imagenet benchmark dataset."""

if __name__ == "__main__":

    import os

    import pycalib.benchmark
    import pycalib.calibration_methods as calm

    ###############################
    #   Generate calibration data
    ###############################

    # Setup
    classify_images = False
    file = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(file)) == "pycalib":
        file += "/datasets/imagenet/"
    else:
        file = os.path.split(os.path.split(file)[0])[0] + "/datasets/imagenet/"
    val_folder = "data/val"
    output_folder = "clf_output"
    n_classes = 1000

    # Classify ImageNet validation data with selected classifiers
    clf_names = [
        'alexnet', 'vgg19',
        'resnet50', 'resnet152',
        'densenet121', 'densenet201',
        'inceptionv4',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
        'polynet',
        "senet154",
        'pnasnet5large',
        'nasnetalarge'
    ]

    if classify_images:
        for clf_name in clf_names:
            pycalib.benchmark.ImageNetData.classify_val_data(file, clf_name=clf_name,
                                                             data_folder=val_folder,
                                                             output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    # Seed and directories
    random_state = 1
    clf_output_dir = os.path.join(file, "clf_output")
    run_dir = os.path.join(file, "calibration")

    # Calibration methods for logits
    cal_methods_logits = {
        "Uncal": calm.NoCalibration(logits=True),
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                      logits=True, random_state=random_state),
        "GPcalib_approx": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                             logits=True, random_state=random_state, inf_mean_approx=True),
        "Temp": calm.TemperatureScaling()
    }

    # Create benchmark object
    imnet_benchmark = pycalib.benchmark.ImageNetData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                     classifier_names=clf_names,
                                                     cal_methods=list(cal_methods_logits.values()),
                                                     cal_method_names=list(cal_methods_logits.keys()),
                                                     use_logits=True, n_splits=10, test_size=10000,
                                                     train_size=1000, random_state=random_state)

    # Run
    imnet_benchmark.run(n_jobs=1)

    # Calibration methods for probability scores
    cal_methods = {
        "Uncal": calm.NoCalibration(),
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10, random_state=random_state),
        # "Temp": calm.TemperatureScaling(),
        "Isotonic": calm.IsotonicRegression(),
        "Beta": calm.BetaCalibration(),
        "Platt": calm.PlattScaling(random_state=random_state),
        "BBQ": calm.BayesianBinningQuantiles()
    }

    # Create benchmark object
    imnet_benchmark = pycalib.benchmark.ImageNetData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                     classifier_names=clf_names,
                                                     cal_methods=list(cal_methods.values()),
                                                     cal_method_names=list(cal_methods.keys()),
                                                     use_logits=False, n_splits=10, test_size=10000,
                                                     train_size=1000, random_state=random_state)

    # Run
    imnet_benchmark.run(n_jobs=1)
