#!/usr/bin/env python
"""aperturesynth - a tool for registering and combining series of photographs.

Usage:
    aperturesynth (--out=OUTPUT) <images>...
  
Options:
    -h --help        Show this screen.
"""

import multiprocessing as mp
from skimage import io, img_as_ubyte, img_as_float
from docopt import docopt

from .register import Registrator
from .gui import get_windows



def save_image(image, filename):
    """Saves the image to the given filename, ensuring uint8 output. """
    io.imsave(filename, img_as_ubyte(image))
    

def load_image(image):
    """Saves the image to the given filename, ensuring uint8 output. """
    return img_as_float(io.imread(image)).astype('float32')


def _transform_worker(registrator, image_queue, transformed_queue):
    """Worker function for multiprocessing image synthesis. """
    init = False
    for image in iter(image_queue.get, 'STOP'):
        image = load_image(image)
        if init:
            acc += registrator(image)[0]
        else:
            acc = registrator(image)[0]
            init = True
    transformed_queue.put(acc)


def process_images(image_list, windows, n_jobs=2):
    """Apply the given transformation to each listed image and find the mean.
    
    Parameters
    ----------
    
    matcher: callable
        Transforms an input array to match the desired baseline image.
    image_list: list of filepaths
        Locations of images to be loaded and transformed. 
    n_workers: int (default=2)
        Number of worker processes to use in parallel.
        
    Returns
    -------
    
    accumulated_image: MxNx[3]
        The registered image as an ndarray. 
        
    """
    # Set up the object to perform the image registration
    baseline = load_image(image_list[0])
    registrator = Registrator(windows, baseline, pad=400)
    
    # Be nice - use half the machine reported cores if n_jobs is not specified.
    if n_jobs == 0:
        n_jobs = int(mp.cpu_count()/2)
        
    if n_jobs == 1:
        for image in image_list[1:]:
            image = load_image(image)
            baseline += registrator(image)[0]
    else:
        image_queue = mp.Queue()
        transformed_queue = mp.Queue()
    
        for image in image_list[1:]:
            image_queue.put(image)
    
        processes = []
        for i in range(n_jobs):
            p = mp.Process(target=_transform_worker,
                           args=(registrator, image_queue, transformed_queue))
            p.start()
            processes.append(p)
            image_queue.put('STOP')
    
        jobs_done = 0
        for transformed in iter(transformed_queue.get, 'DUMMY'):
            baseline += transformed
            jobs_done += 1
            if jobs_done == n_jobs:
                break

        for p in processes:
            p.join()
            
    baseline /= len(image_list)
    return baseline

def main():
    """Registers and transforms each input image and saves the result."""
    args = docopt(__doc__)
    images = args['<images>']
    output_file = args['--out']
    
    windows = get_windows(load_image(images[0]))

    output = process_images(images, windows)

    save_image(output, output_file)
