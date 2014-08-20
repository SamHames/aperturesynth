import numpy as np
import matplotlib.pyplot as plt 


def get_windows(image):
    """ display the image, user selects points and processes the output to a series
    of window coordinates"""
    plt.interactive(True)
    plt.imshow(image); plt.show()
    crop = plt.ginput(0)
    plt.close()
    plt.interactive(False)
    # remove last point if an odd number selected
    crop = crop[:-1] if np.mod(len(crop),2) else crop
    return np.vstack(crop).astype('int')[:,[1,0]]