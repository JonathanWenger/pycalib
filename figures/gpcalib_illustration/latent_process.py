"""Plot the latent function of GP calibration on a synthetic data set."""

if __name__ == "__main__":

    import numpy as np
    import os

    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm

    # Setup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(dir_path)) == "pycalib":
        dir_path += "/figures/gpcalib_illustration/"

    # Seed
    seed = 0

    # Generate synthetic classification data
    def miscal(z):
        """Miscalibration function."""
        return z ** 3

    n_classes = 4
    Z, y, info_dict = bm.SyntheticBeta.sample_miscal_data(alpha=2, beta=.75, miscal_func=miscal,
                                                          miscal_func_name="power",
                                                          size=100, marginal_probs=np.ones(n_classes) / n_classes,
                                                          random_state=seed)

    # Calibrate
    gpc = calm.GPCalibration(n_classes=n_classes,
                             maxiter=1000,
                             n_inducing_points=10,
                             verbose=True)
    gpc.fit(Z, y)

    # Plot
    gpc.plot_latent(
        z=np.linspace(start=.001, stop=1, num=1000),
        filename=dir_path + "latent_process",
        plot_classes=True,
        ratio=0.5,
        gridspec_kw={'height_ratios': [3, 2]})
