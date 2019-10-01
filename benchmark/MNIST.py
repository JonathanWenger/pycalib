import os.path
import numpy as np
import gpflow
import xgboost as xgb

# Install latest version of scikit-garden from github to enable partial_fit(X, y):
# (https://github.com/scikit-garden/scikit-garden)
from skgarden import MondrianForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import pycalib.benchmark
import pycalib.calibration_methods as calm

###############################
#   Generate calibration data
###############################

if __name__ == "__main__":

    # Classify MNIST validation data with selected classifiers
    random_state = 1

    clf_dict = {
        "AdaBoost": AdaBoostClassifier(base_estimator=None,
                                       n_estimators=100,
                                       learning_rate=1.0,
                                       random_state=random_state),
        "XGBoost": xgb.XGBClassifier(booster="gbtree",
                                     n_estimators=100,
                                     random_state=random_state,
                                     n_jobs=-1),
        "mondrian_forest": MondrianForestClassifier(n_estimators=10,
                                                    min_samples_split=2,
                                                    bootstrap=False,
                                                    n_jobs=-1,
                                                    random_state=random_state,
                                                    verbose=0),
        "random_forest": RandomForestClassifier(n_estimators=100,
                                                criterion="gini",
                                                min_samples_split=2,
                                                bootstrap=True,
                                                n_jobs=-1,
                                                random_state=random_state),
        "1layer_NN": MLPClassifier(hidden_layer_sizes=(100,),
                                   activation="relu",
                                   solver="adam",
                                   random_state=random_state)
    }

    # Setup
    file = "/home/j/Documents/research/projects/nonparametric_calibration/pycalib/datasets/mnist/"
    output_folder = "clf_output"
    classify_images = False

    if classify_images:
        for clf_name, clf in clf_dict.items():
            pycalib.benchmark.MNISTData.classify_val_data(file, clf_name=clf_name, classifier=clf,
                                                          output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    # Initialization
    run_dir = os.path.join(file, "calibration")
    data_dir = os.path.join(file, "clf_output")

    # Classifiers
    classifier_names = list(clf_dict.keys())

    # Calibration models
    with gpflow.defer_build():
        meanfunc = pycalib.gp_classes.ScalarMult()
        meanfunc.alpha.transform = gpflow.transforms.positive
    cal_methods = {
        "Uncal": calm.NoCalibration(),
        "GPcalib": calm.GPCalibration(n_classes=10, maxiter=1000, n_inducing_points=10, logits=False,
                                      random_state=random_state),
        "GPcalib_lin": calm.GPCalibration(n_classes=10, maxiter=1000, mean_function=meanfunc,
                                          n_inducing_points=10, logits=False,
                                          random_state=random_state),
        "GPcalib_approx": calm.GPCalibration(n_classes=10, maxiter=1000, n_inducing_points=10,
                                             logits=False, random_state=random_state, inf_mean_approx=True),
        "Platt": calm.PlattScaling(random_state=random_state),
        "Isotonic": calm.IsotonicRegression(),
        "Beta": calm.BetaCalibration(),
        "BBQ": calm.BayesianBinningQuantiles(),
        "Temp": calm.TemperatureScaling()
    }


    # Create benchmark object
    mnist_benchmark = pycalib.benchmark.MNISTData(run_dir=run_dir, data_dir=data_dir,
                                                  classifier_names=classifier_names,
                                                  cal_methods=list(cal_methods.values()),
                                                  cal_method_names=list(cal_methods.keys()),
                                                  n_splits=10, test_size=9000,
                                                  train_size=1000, random_state=random_state)

    # Run
    mnist_benchmark.run(n_jobs=1)
