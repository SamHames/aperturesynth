"""aperturesynth - a tool for registering and combining series of photographs.

Usage:
    aperturesynth [--no-transform] [--out FILE] <images>...

Options:
    -h --help           Show this help screen.
    --out FILE          Optional output file. If not specified the output will
                        be written to a tiff file with same name as the
                        baseline image with 'transformed_' prepended.
    --no-transform      Combine images without transforming first. Useful for
                        visualising the impact of registration.

The first image passed in will be the baseline image to which all following
images will be matched.

"""


import multiprocessing as mp
from skimage import io, img_as_ubyte, img_as_float
from docopt import docopt
import os.path

from .register import Registrator
from .gui import get_windows


def save_image(image, filename):
    """Saves the image to the given filename, ensuring uint8 output. """
    io.imsave(filename, img_as_ubyte(image))


def load_image(image):
    """Loads the given file and converts to float32 format. """
    return img_as_float(io.imread(image)).astype('float32')


def process_images(image_list, windows, no_transform=False):
    """Apply the given transformation to each listed image and find the mean.

    Parameters
    ----------

    image_list: list of filepaths
        Locations of images to be loaded and transformed.
    windows:
    n_jobs: int (default 1)
        Number of worker processes to use in parallel.
    no_transform: bool (default False)
        If true, combine images without registering them first. The windows
        and n_jobs variables will be ignored. Useful for visualising the impact
        of the registration process.

    Returns
    -------

    image: MxNx[3]
        The combined image as an ndarray.

    """
    if no_transform:
        baseline = load_image(image_list[0])
        for image in image_list[1:]:
            baseline += load_image(image)
    else:
        # Set up the object to perform the image registration
        baseline = load_image(image_list[0])
        registrator = Registrator(windows, baseline, pad=400)

        for image in image_list[1:]:
            image = load_image(image)
            baseline += registrator(image)[0]

    baseline /= len(image_list)
    return baseline


def main():
    """Registers and transforms each input image and saves the result."""
    args = docopt(__doc__)
    images = args['<images>']

    if args['--out'] is not None:
        output_file = args['--out']
    else:
        head, ext = os.path.splitext(images[0])
        head, tail = os.path.split(head)
        output_file = os.path.join(head, 'transformed_' + tail + '.tiff')

    if args['--no-transform']:
        windows = []
        output = process_images(images, windows, no_transform=True)
        save_image(output, output_file)
    else:
        windows = get_windows(load_image(images[0]))
        output = process_images(images, windows)
        save_image(output, output_file)
