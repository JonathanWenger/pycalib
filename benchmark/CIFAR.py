import os.path
import gpflow
import pycalib.benchmark
import pycalib.calibration_methods as calm

###############################
#   Generate calibration data
###############################

if __name__ == "__main__":

    # Classify CIFAR-100 validation data with selected classifiers
    clf_names = [
        'alexnet',
        'vgg19',
        'resnet110',
        'resnext-8x64d',
        'resnext-16x64d',
        'densenet-bc-100-12',
        'densenet-bc-L190-k40'
    ]

    # Setup
    file = "/home/j/Documents/research/projects/nonparametric_calibration/pycalib/data/cifar/"
    val_folder = "val"
    output_folder = "clf_output"
    classify_images = False

    if classify_images:
        for clf_name in clf_names:
            pycalib.benchmark.CIFARData.classify_val_data(file, clf_name=clf_name,
                                                          validation_folder=val_folder,
                                                          output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    # Seed and directories
    random_state = 1
    data_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")
    n_classes = 100

    # Calibration methods for logits
    with gpflow.defer_build():
        meanfunc = pycalib.gp_classes.ScalarMult()
        meanfunc.alpha.transform = gpflow.transforms.positive

    cal_methods_logits = {
        "Uncal": calm.NoCalibration(logits=True),
        "GPcalib_lin": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                          mean_function=meanfunc, logits=True, verbose=False,
                                          random_state=random_state),
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                      logits=True, random_state=random_state),
        "GPcalib_approx": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
                                             logits=True, random_state=random_state, inf_mean_approx=True),
        "Temp": calm.TemperatureScaling()
    }

    # Create benchmark object
    cifar_benchmark = pycalib.benchmark.CIFARData(run_dir=run_dir, data_dir=data_dir,
                                                  classifier_names=clf_names,
                                                  cal_methods=list(cal_methods_logits.values()),
                                                  cal_method_names=list(cal_methods_logits.keys()),
                                                  use_logits=True, n_splits=10, test_size=10000,
                                                  train_size=1000, random_state=random_state)


    # Run
    cifar_benchmark.run(n_jobs=1)

    # Calibration
    cal_methods = {
        "Uncal": calm.NoCalibration(),
        "GPcalib": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10, random_state=random_state),
        # "GPcalib_lin": calm.GPCalibration(n_classes=n_classes, maxiter=1000, n_inducing_points=10,
        #                                   mean_function=meanfunc, logits=False, verbose=False,
        #                                   random_state=random_state),
        # "Temp": calm.TemperatureScaling(),
        "Isotonic": calm.IsotonicRegression(),
        "Beta": calm.BetaCalibration(),
        "Platt": calm.PlattScaling(random_state=random_state),
        "BBQ": calm.BayesianBinningQuantiles()
    }

    # Create benchmark object
    cifar_benchmark = pycalib.benchmark.CIFARData(run_dir=run_dir, data_dir=data_dir,
                                                  classifier_names=clf_names,
                                                  cal_methods=list(cal_methods.values()),
                                                  cal_method_names=list(cal_methods.keys()),
                                                  use_logits=False, n_splits=10, test_size=10000,
                                                  train_size=1000, random_state=random_state)

    # Run
    cifar_benchmark.run(n_jobs=1)
