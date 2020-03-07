"""Calibration experiment on the CIFAR-100 benchmark dataset."""

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
        file += "/datasets/cifar100/"
    else:
        file = os.path.split(os.path.split(file)[0])[0] + "/datasets/cifar100/"
    data_folder = "data"
    output_folder = "clf_output"

    # Classify CIFAR-100 validation data with selected classifiers
    clf_names = [
        'alexnet',
        'WRN-28-10-drop'
        'resnext-8x64d',
        'resnext-16x64d',
        'densenet-bc-l190-k40',
    ]

    # Classify images in data set
    if classify_images:
        for clf_name in clf_names:
            pycalib.benchmark.CIFARData.classify_val_data(dataset_folder=file, clf_name=clf_name,
                                                          data_folder=data_folder,
                                                          output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    # Seed and directories
    random_state = 1
    clf_output_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")
    n_classes = 100
    train_size = 1000
    test_size = 9000

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
    cifar_benchmark = pycalib.benchmark.CIFARData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                  classifier_names=clf_names,
                                                  cal_methods=list(cal_methods_logits.values()),
                                                  cal_method_names=list(cal_methods_logits.keys()),
                                                  use_logits=True, n_splits=10, test_size=test_size,
                                                  train_size=train_size, random_state=random_state)

    # Run
    cifar_benchmark.run(n_jobs=1)

    # Calibration methods for probability scores
    cal_methods = {
        "Uncal": calm.NoCalibration(),
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                      random_state=random_state),
        # "Temp": calm.TemperatureScaling(),
        "Isotonic": calm.IsotonicRegression(),
        "Beta": calm.BetaCalibration(),
        "Platt": calm.PlattScaling(random_state=random_state),
        "BBQ": calm.BayesianBinningQuantiles()
    }

    # Create benchmark object
    cifar_benchmark = pycalib.benchmark.CIFARData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                  classifier_names=clf_names,
                                                  cal_methods=list(cal_methods.values()),
                                                  cal_method_names=list(cal_methods.keys()),
                                                  use_logits=False, n_splits=10, test_size=test_size,
                                                  train_size=train_size, random_state=random_state)

    # Run
    cifar_benchmark.run(n_jobs=1)
