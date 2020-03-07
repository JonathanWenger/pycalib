"""Train a simple neural network (and calibrate post-hoc) while monitoring log-loss and calibration error."""

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import os

    from sklearn.datasets import fetch_openml
    from sklearn import model_selection as ms
    from sklearn.neural_network import MLPClassifier
    import sklearn.metrics as metr

    import pycalib.texfig as texfig
    import pycalib.scoring as meas
    import pycalib.calibration_methods as calm

    # Setup filepath
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(dir_path)) == "pycalib":
        dir_path += "/figures/ece_logloss_example"

    # Seed
    random_state = 42

    # Initialization and folder structure
    reliability_path = os.path.join(dir_path, "reliability_diagrams")
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(reliability_path, exist_ok=True)

    metrics_test = {"ECE": [],
                    "NLL": [],
                    "err": [],
                    "iter": []
                    }
    metrics_train = {"ECE": [],
                     "NLL": [],
                     "err": [],
                     "iter": []
                     }

    # Load MNIST data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=False)
    X = X / 255.
    y = np.array(y, dtype=int)

    # Sample training/calibration and test set
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=11000, test_size=50000,
                                                           random_state=random_state)
    X_train, X_cal, y_train, y_cal = ms.train_test_split(X_train, y_train, train_size=10000, test_size=1000,
                                                         random_state=random_state)

    # Define vanilla neural network
    mlp = MLPClassifier(hidden_layer_sizes=(100,),
                        solver='adam',
                        verbose=True,
                        max_iter=1,
                        warm_start=True,
                        random_state=random_state)

    for iter in np.arange(start=1, stop=66, step=1):
        # Train
        mlp.fit(X_train, y_train)

        # Predict
        p_pred_train = mlp.predict_proba(X_train)
        y_pred_train = np.argmax(p_pred_train, axis=1)
        p_pred_test = mlp.predict_proba(X_test)
        y_pred_test = np.argmax(p_pred_test, axis=1)

        # Evaluate metrics
        metrics_test["ECE"].append(meas.expected_calibration_error(y=y_test, p_pred=p_pred_test, n_classes=10))
        metrics_test["NLL"].append(metr.log_loss(y_true=y_test, y_pred=p_pred_test))
        metrics_test["err"].append(1 - metr.accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=True))
        metrics_test["iter"].append(mlp.n_iter_)

        metrics_train["ECE"].append(meas.expected_calibration_error(y=y_train, p_pred=p_pred_train, n_classes=10))
        metrics_train["NLL"].append(metr.log_loss(y_true=y_train, y_pred=p_pred_train))
        metrics_train["err"].append(1 - metr.accuracy_score(y_true=y_train, y_pred=y_pred_train, normalize=True))
        metrics_train["iter"].append(mlp.n_iter_)

    # Calibration
    gpc = calm.GPCalibration(n_classes=len(np.unique(y)), random_state=random_state)
    gpc.fit(mlp.predict_proba(X_cal), y_cal)
    gpc.plot_latent(filename=os.path.join(dir_path, "gpc_latent"), z=np.linspace(start=10 ** -3, stop=1, num=1000))
    p_calib = gpc.predict_proba(p_pred_test)
    ece_calib = meas.expected_calibration_error(y=y_test, p_pred=p_calib)
    acc_calib = meas.accuracy(y=y_test, p_pred=p_calib)
    nll_calib = metr.log_loss(y_true=y_test, y_pred=p_calib)

    # Save data to file
    json.dump(metrics_train, open(os.path.join(dir_path, "metrics_train.txt"), 'w'))
    json.dump(metrics_test, open(os.path.join(dir_path, "metrics_test.txt"), 'w'))
    json.dump({"ECE_calib": ece_calib,
               "acc_calib": acc_calib,
               "NLL_calib": nll_calib}, open(os.path.join(dir_path, "metrics_calib.txt"), 'w'))

    # Load data from file
    metrics_train = json.load(open(os.path.join(dir_path, "metrics_train.txt")))
    metrics_test = json.load(open(os.path.join(dir_path, "metrics_test.txt")))
    metrics_calib = json.load(open(os.path.join(dir_path, "metrics_calib.txt")))

    # Plot accuracy, ECE, logloss
    var_list = [
        ("err", "error"),
        ("NLL", "NLL"),
        ("ECE", "$\\textup{ECE}_1$")
    ]

    fig, axes = texfig.subplots(width=7, ratio=.2, nrows=1, ncols=3, w_pad=1)

    for i, (m, lab) in enumerate(var_list):
        axes[i].plot(metrics_train["iter"][2::], metrics_train[m][2::], label="train")
        axes[i].plot(metrics_test["iter"][2::], metrics_test[m][2::], label="test")
        axes[i].set_xlabel("epoch")
        axes[i].set_ylabel(lab)

    for j, metric_calib in enumerate(
            [1 - metrics_calib["acc_calib"], metrics_calib["NLL_calib"], metrics_calib["ECE_calib"]]):
        axes[j].axhline(y=metric_calib, xmin=0, xmax=1, color="red", linestyle="--")
        axes[j].plot(metrics_test["iter"][-1], metric_calib, label="test + calibr.",
                     color="red", marker='o', markersize=6)
    axes[2].legend(prop={'size': 9}, labelspacing=0.2)
    texfig.savefig(os.path.join(dir_path, "ece_logloss"))
    plt.close('all')
