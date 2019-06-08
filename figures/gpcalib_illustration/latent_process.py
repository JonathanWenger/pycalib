if __name__ == "__main__":

    import numpy as np
    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm

    # Synthetic data
    def f(x):
        return x ** 3


    n_classes = 4
    X, y, info_dict = bm.SyntheticBeta.sample_miscal_data(alpha=2, beta=.75, miscal_func=f, miscal_func_name="power",
                                                          size=100, marginal_probs=np.ones(n_classes) / n_classes,
                                                          random_state=0)

    # Calibrate
    gpc = calm.GPCalibration(n_classes=n_classes,
                             maxiter=1000,
                             n_inducing_points=10,
                             verbose=True)
    gpc.fit(X, y)

    # Plot
    file = "/Users/jwenger/Documents/university/theses/master's thesis/code/" +\
           "pycalib/pycalib/figures/gpcalib_illustration/latent_process"
    gpc.plot(filename=file,
             plot_classes=True,
             xlim=[10**-3, 1],
             ratio=0.5,
             gridspec_kw={'height_ratios': [3, 2]})
