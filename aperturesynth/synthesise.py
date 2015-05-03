"""aperturesynth - a tool for registering and combining series of photographs.

Usage:
    aperturesynth combine [--no-transform] [--out FILE] <images>...
    aperturesynth choose_windows <base_image> <window_file>

Options:
    -h --help           Show this help screen.
    --out FILE          Optional output file. If not specified the output will
                        be written to a tiff file with same name as the
                        baseline image with 'transformed_' prepended. The
                        output format is chosen by the file extension.
    --no-transform      Combine images without transforming first. Useful for
                        visualising the impact of registration.

The first image passed in will be the baseline image to which all following
images will be matched.

"""


import multiprocessing as mp
import numpy as np
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


def register_images(image_list, registrator):
    """A generator to register a series of images.

    The first image is taken as the baseline and is not transformed.

    """
    yield load_image(image_list[0])

    for image_file in image_list[1:]:
        transformed_image, transform = registrator(load_image(image_file))
        # Stub for future operations that examine the transformation
        yield transformed_image


def no_transform(image):
    """Pass through the original image without transformation.

    Returns a tuple with None to maintain compatability with processes that
    evaluate the transform.

    """
    return (image, None)


def process_images(image_list, registrator, fusion=None):
    """Apply the given transformation to each listed image and find the mean.

    Parameters
    ----------

    image_list: list of filepaths
        Image files to be loaded and transformed.
    registrator: callable
        Returns the desired transformation of a given image.
    fusion: callable (optional, default=None)
        Returns the fusion of the given images. If not specified the images are
        combined by averaging.

    Returns
    -------

    image: MxNx[3]
        The combined image as an ndarray.

    """

    registered = register_images(image_list, registrator)

    if fusion is not None: # Stub for future alternative fusion methods
        return fusion(registered)

    else:
        output = sum(registered)
        output /= len(image_list)
        return output


def main():
    """Registers and transforms each input image and saves the result."""
    args = docopt(__doc__)

    if args['choose_windows'] is not None:
        print(args)
        reference = load_image(args['<base_image>'])
        windows = get_windows(reference)
        np.savetxt(args['<window_file>'], windows.astype('int'), fmt='%i')

    else:
        images = args['<images>']

        if args['--out'] is not None:
            output_file = args['--out']
        else:
            head, ext = os.path.splitext(images[0])
            head, tail = os.path.split(head)
            output_file = os.path.join(head, 'transformed_' + tail + '.tiff')

        if args['--no-transform']:
            registrator = no_transform
        else:
            baseline = load_image(images[0])
            windows = get_windows(baseline)
            registrator = Registrator(windows, baseline)

        output = process_images(images, registrator)
        save_image(output, output_file)
