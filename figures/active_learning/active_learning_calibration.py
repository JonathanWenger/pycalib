import os
import numpy as np
# Install latest version of scikit-garden from github to enable partial_fit(X, y):
# (https://github.com/scikit-garden/scikit-garden)
from skgarden import MondrianForestClassifier
import pycalib.calibration_methods as calib
import pycalib.active_learning as al
import gpflow
import pycalib

if __name__ == "__main__":
    # Setup
    random_state = 0

    # Load KITTI data
    data_dir = "/Users/jwenger/Documents/university/theses/master's thesis/code/pycalib/data/kitti"
    files = ['kitti_all_train.data',
             'kitti_all_train.labels',
             'kitti_all_test.data',
             'kitti_all_test.labels']
    X_train = np.loadtxt(os.path.join(data_dir, files[0]), np.float64, skiprows=1)
    y_train = np.loadtxt(os.path.join(data_dir, files[1]), np.int32, skiprows=1)
    X_test = np.loadtxt(os.path.join(data_dir, files[2]), np.float64, skiprows=1)
    y_test = np.loadtxt(os.path.join(data_dir, files[3]), np.int32, skiprows=1)

    # Training and active learning parameters
    n_classes = len(np.unique(np.hstack([y_train, y_test])))
    calib_method_name = "GPcalib"

    mf = MondrianForestClassifier(n_estimators=10,
                                  min_samples_split=2,
                                  bootstrap=False,
                                  n_jobs=1,
                                  random_state=random_state,
                                  verbose=0)

    # Active learning
    al_exp = al.ActiveLearningExperiment(classifier=mf, X_train=X_train,
                                         y_train=y_train,
                                         X_test=X_test,
                                         y_test=y_test,
                                         query_criterion=al.query_entropy,
                                         uncertainty_thresh=0.5,
                                         pretrain_size=1000,
                                         calibration_method=calib.GPCalibration(n_classes=n_classes, maxiter=1000,
                                                                                n_inducing_points=10, verbose=False,
                                                                                random_state=random_state),
                                         # calibration_method=calib.TemperatureScaling(),
                                         calib_size=200,
                                         calib_points=[1000, 2000, 3000, 4000],
                                         batch_size=250)

    result_df = al_exp.run(n_cv=10, random_state=random_state)

    # Save to file
    dir = "/Users/jwenger/Documents/university/theses/master's thesis/code/pycalib/pycalib/figures/active_learning"
    al_exp.save_result(file=dir)

    al_exp.result_df = al_exp.load_result(file=dir + "/active_learning_results.csv")

    # Plot
    plot_dict = {"AL_ECE_error": ["ECE", "error"],
                 "AL_over_underconfidence": ["overconfidence", "underconfidence"],
                 # "AL_ECE_sharpness": ["ECE", "sharpness"]
                 }

    for filename, metrics_list in plot_dict.items():
        al_exp.plot(file=os.path.join(dir, filename), metrics_list=metrics_list, scatter=True, confidence=False)
