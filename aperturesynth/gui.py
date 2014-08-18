import numpy
import matplotlib.pyplot as plt 
from PIL import Image

def get_windows(image):
    """ display the image, user selects points and processes the output to a series
    of window coordinates"""
    plt.interactive(True)
    im = numpy.asfarray(Image.open(image))
    plt.imshow(im/255.); plt.show()
    crop = plt.ginput(0)
    plt.close()
    plt.interactive(False)
    # remove last point if an odd number selected
    crop = crop[:-1] if numpy.mod(len(crop),2) else crop
    return numpy.vstack(crop).astype('int')[:,[1,0]]