if __name__ == "__main__":
    import numpy as np
    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm


    # Synthetic data
    def miscal(z):
        return z ** 3


    n_classes = 4
    Z, y, info_dict = bm.SyntheticBeta.sample_miscal_data(alpha=2, beta=.75, miscal_func=miscal, miscal_func_name="power",
                                                          size=100, marginal_probs=np.ones(n_classes) / n_classes,
                                                          random_state=0)

    # Calibrate
    gpc = calm.GPCalibration(n_classes=n_classes,
                             maxiter=1000,
                             n_inducing_points=10,
                             verbose=True)
    gpc.fit(Z, y)

    # Plot
    file = "/home/j/Documents/research/projects/nonparametric_calibration/" + \
           "pycalib/figures/gpcalib_illustration/latent_process"
    gpc.plot_latent(
        z=np.linspace(start=.001, stop=1, num=1000),
        filename=file,
        plot_classes=True,
        ratio=0.5,
        gridspec_kw={'height_ratios': [3, 2]})
