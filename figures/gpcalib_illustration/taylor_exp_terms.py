if __name__ == "__main__":
    import numpy as np
    import pycalib.texfig as texfig
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.special import logsumexp, softmax


    # Log-softargmax function being approximated
    def logsoftargmax(f, y):
        return f[:, y] - logsumexp(f, axis=1)


    def taylor_logsoftargmax(f, y, loc):
        e_y = np.zeros(np.shape(f)[1])
        e_y[y] = 1
        sigma = softmax(loc)
        grad = e_y - sigma
        hessian = np.einsum('n, k -> nk', sigma, sigma) - np.diag(sigma)
        return np.log(np.matmul(e_y, sigma)) + np.matmul(f - loc, grad) + 0.5 * np.einsum('nd, dd, nd -> n', f - loc,
                                                                                          hessian, f - loc)


    # Data
    y = 0
    f1, f2 = np.meshgrid(np.arange(-2, 2, .1), np.arange(-2, 2, .1), sparse=False)
    f = np.column_stack([np.reshape(f1, -1, 1), np.reshape(f2, -1, 1)])
    phi = np.array([0, 0])

    h = np.reshape(logsoftargmax(f, y=0), np.shape(f1))  # Function evaluated at inputs
    taylor_h = np.reshape(taylor_logsoftargmax(f, y=0, loc=phi), np.shape(f1))

    # Meshplot
    fig = texfig.figure(width=6)
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X=f1, Y=f2, Z=h, label="$\log p(y_n \mid f_n)$", color="tab:blue")
    ax.plot_wireframe(X=f1, Y=f2, Z=taylor_h, label="Taylor approx.", color="tab:orange")

    ax.set_xlabel("$f_n^1$")
    ax.set_ylabel("$f_n^2$")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()
    # ax.legend(prop={'size': 9}, labelspacing=0.2)
    texfig.savefig("figures/gpcalib_illustration/taylor_approx_mesh")

    # Contour difference plot
    fig, axes = texfig.subplots()
    c = axes.contourf(f1, f2, h - taylor_h)
    fig.colorbar(c, ax=axes)
    axes.set_xlabel("$f_n^1$")
    axes.set_ylabel("$f_n^2$")

    texfig.savefig("figures/gpcalib_illustration/taylor_approx_contour")
    plt.close("all")
