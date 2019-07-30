if __name__ == "__main__":
    import numpy as np
    import pycalib.calibration_methods as calm
    import pycalib.benchmark as bm
    import pycalib.texfig as texfig


    # Synthetic data
    def miscal(z):
        return z ** 3


    n_classes = 4
    Z, y, info_dict = bm.SyntheticBeta.sample_miscal_data(alpha=2, beta=.75, miscal_func=miscal, miscal_func_name="power",
                                                          size=100, marginal_probs=np.ones(n_classes) / n_classes,
                                                          random_state=0)

    # Calibrate
    ts = calm.TemperatureScaling()
    ts.fit(Z, y)

    gpc = calm.GPCalibration(n_classes=n_classes,
                             maxiter=1000,
                             n_inducing_points=10,
                             verbose=True)
    gpc.fit(Z, y)

    # Get latent function values
    z = np.linspace(start=.001, stop=1, num=1000)
    ts_latent = ts.latent(z)
    gpc_latent, gpc_latent_var = gpc.latent(z)

    # Plot
    filename = "/Users/jwenger/Documents/research/projects/nonparametric_calibration/code/" + \
           "pycalib/figures/latent_functions/synthetic_data"

    fig, axes = texfig.subplots(nrows=1, ncols=1, sharex=True)
    axes.plot(z, ts.T*z, label="Temp. Scaling")
    axes.plot(z, gpc_latent, label="GPcalib")
    axes.fill_between(z, gpc_latent - 2 * np.sqrt(gpc_latent_var), gpc_latent + 2 * np.sqrt(gpc_latent_var), alpha=.2)
    axes.set_xlabel("$z$")
    axes.set_ylabel("$f(z)$")

    # Save plot to file
    texfig.savefig(filename)
