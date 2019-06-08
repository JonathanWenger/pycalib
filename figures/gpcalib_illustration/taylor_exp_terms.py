if __name__ == "__main__":
    import numpy as np
    import pycalib.texfig as texfig
    import matplotlib.pyplot as plt
    from scipy.special import logsumexp, softmax
    from scipy.interpolate import approximate_taylor_polynomial


    # Log-softargmax function being approximated
    def logsoftargmax(f, y):
        return f[:, y] - logsumexp(f, axis=1)


    def taylor_logsoftargmax(f, y, loc):
        e_y = np.zeros(np.shape(f)[1])
        e_y[y] = 1
        sigma = softmax(loc)
        grad = e_y - sigma
        hessian = np.einsum('n, k -> nk', sigma, sigma) - np.diag(sigma)
        return np.matmul(e_y, sigma) + np.matmul(f - loc, grad) + np.einsum('nd, dd, nd -> n', f - loc, hessian,
                                                                            f - loc)


    # Data
    y = 0
    f1, f2 = np.meshgrid(np.arange(-10, 10, .01), np.arange(-10, 10, .01), sparse=False)
    f = np.column_stack([np.reshape(f1, -1, 1), np.reshape(f2, -1, 1)])

    h = np.reshape(logsoftargmax(f, y=0), np.shape(f1))  # Function evaluated at inputs

    # Plot
    fig, axes = texfig.subplots(width=7, ratio=.2, nrows=1, ncols=2, w_pad=1)

    # Meshplot

    # Contour difference plot
    levels = np.linspace(np.min(h), np.max(h), 5)
    axes[1].contourf(X=f1, Y=f2, Z=h, levels=levels)

    # axes[1].legend(prop={'size': 9}, labelspacing=0.2)
    texfig.savefig("figures/gpcalib_illustration/taylor_approx")
    plt.close("all")
