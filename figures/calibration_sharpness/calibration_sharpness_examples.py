if __name__ == "__main__":

    import numpy as np
    from pycalib.benchmark import SyntheticBeta as sb
    import pycalib.texfig as texfig
    from pycalib.plotting import reliability_diagram

    # Generate synthetic data
    seed = 0
    n_samples = 10000
    n_classes = 3
    marg_probs = np.ones(n_classes) / n_classes
    n_bins = 15

    # Miscalibrated and not sharp
    def f(x):
        return x ** 3


    X_miscal_notsharp, y_miscal_notsharp, _ = sb.sample_miscal_data(alpha=2, beta=2, miscal_func=f,
                                                                    miscal_func_name="power",
                                                                    size=n_samples, marginal_probs=marg_probs,
                                                                    random_state=seed)
    reliability_diagram(y_miscal_notsharp, X_miscal_notsharp,
                        filename="figures/calibration_sharpness/miscal_notsharp",
                        title=None, n_bins=n_bins)

    # Calibrated and not sharp
    def id(x):
        return x


    X_cal_notsharp, y_cal_notsharp, _ = sb.sample_miscal_data(alpha=3, beta=3, miscal_func=id,
                                                              miscal_func_name="id",
                                                              size=n_samples, marginal_probs=marg_probs,
                                                              random_state=seed)
    reliability_diagram(y_cal_notsharp, X_cal_notsharp,
                        filename="figures/calibration_sharpness/cal_notsharp",
                        title=None, n_bins=n_bins)

    # Miscalibrated and sharp
    X_miscal_sharp, y_miscal_sharp, _ = sb.sample_miscal_data(alpha=.5, beta=.5, miscal_func=f,
                                                              miscal_func_name="power",
                                                              size=n_samples, marginal_probs=marg_probs,
                                                              random_state=seed)
    reliability_diagram(y_miscal_sharp, X_miscal_sharp,
                        filename="figures/calibration_sharpness/miscal_sharp",
                        title=None, n_bins=n_bins)

    # Calibrated and sharp
    X_cal_sharp, y_cal_sharp, _ = sb.sample_miscal_data(alpha=.5, beta=.5, miscal_func=id,
                                                        miscal_func_name="id",
                                                        size=n_samples, marginal_probs=marg_probs,
                                                        random_state=seed)
    reliability_diagram(y_cal_sharp, X_cal_sharp,
                        filename="figures/calibration_sharpness/cal_sharp",
                        title=None, n_bins=n_bins)

