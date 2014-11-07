import numpy as np
import matplotlib.pyplot as plt


def get_windows(image):
    """Display the given image and record user selected points.

    Parameters
    ----------

    image : NxMx3 ndarray
        The image to be displayed.

    Returns
    -------

    array : n_points x 2
        An array of coordinates in the image. Each row corresponds to the x, y
        coordinates of one point. If an odd number of points are specified, the
        last one will be discarded.

    """
    plt.interactive(True)
    plt.imshow(image)
    plt.show()
    crop = plt.ginput(0)
    plt.close()
    plt.interactive(False)
    # remove last point if an odd number selected
    crop = crop[:-1] if np.mod(len(crop), 2) else crop
    return np.vstack(crop).astype('int')[:, [1, 0]]
