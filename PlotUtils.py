def finalPlot(P, finalX, finalY, image=None, contour=None, n=300):
    # plot final grasping point representation.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot: matplotlib was not installed.")
        return

    if contour is not None:
        plt.plot(contour[:, 0], contour[:, 1], 'c--', linewidth=2)
    plt.plot(P[:, 0], P[:, 1], 'y', linewidth=2)
    if image is not None:
        plt.imshow(image, plt.cm.gray)
    # plt.show()
    plt.plot(P[finalX, 0], P[finalX, 1], 'r+', linewidth=2)
    plt.plot(P[finalY, 0], P[finalY, 1], 'r+', linewidth=2)
    # plt.show(block=True)
    plt.pause(0.01)
    plt.show(block=True)


def plot_efd(P, N, Cbar, image=None, contour=None, n=200):

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot: matplotlib was not installed.")
        return

    # print(contour[:,1],contour[:,0])

    # for ii in range(1, len(xt)):
    #     plt.plot(xt[ii], yt[ii], 'r*', linewidth=2)
    #     plt.show()

    # plt.set_title(str(n + 1))
    if contour is not None:
        plt.plot(contour[:, 0], contour[:, 1], 'c--', linewidth=2)
    plt.plot(P[:, 0], P[:, 1], 'r', linewidth=2)
    if image is not None:
        plt.imshow(image, plt.cm.gray)
    # plt.show()

    for ii in range(1, n):
        if Cbar[ii] > 0:
            plt.plot(P[ii, 0], P[ii, 1], 'y*', linewidth=2)
    plt.show()
