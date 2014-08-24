import multiprocessing as mp
from skimage import io, img_as_float, img_as_ubyte


def save_image(image, filename):
    """Saves the image to the given filename, ensuring unit8 output. """
    io.imsave(filename, img_as_ubyte(image))


def load_image(filename):
    """Loads the given image file to a floating point ndarray. """
    return img_as_float(io.imread(filename))


def _transform_worker(matcher, image_queue, transformed_queue):
    """Worker function for multiprocessing image synthesis. """
    for image in iter(image_queue.get, 'STOP'):
        image = load_image(image)
        try:
            acc += matcher(image)[0]
        except NameError:
            acc = matcher(image)[0]
    transformed_queue.put(acc)


def process_images(matcher, image_list, n_workers=2):
    """Apply the given transformation to a list of images.
    
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
    image_queue = mp.Queue()
    accumulate_queue = mp.Queue()

    for image in image_list[1:]:
        image_queue.put(image)

    if n_workers == 0:
        n_workers = int(mp.cpu_count()/2)  # Hack to account for hyperthreading, and also not consume all available resources

    processes = []
    # start a series of workers for each process
    for i in range(n_workers):
        p = mp.Process(target=_transform_worker,
                       args=(matcher, image_queue, accumulate_queue))
        p.start()
        processes.append(p)
        image_queue.put('STOP')

    procs_done = 0
    acc = load_image(image)
    for accumulated in iter(accumulate_queue.get, 'DUMMY'):
        acc += accumulated
        procs_done += 1
        if procs_done == n_workers:
            break
    for p in processes:
        p.join()
    acc /= len(image_list)
    return acc


if __name__ == "__main__":
    from glob import glob
    from register import Registrator
    from gui import get_windows
    import matplotlib.pyplot as plt

    images = glob('/home/sam/photos/computational/bulkregister/P106031[5-9]*')

    base = load_image(images[0])

    windows = get_windows(base)
    matcher = Registrator(windows, base, pad=400)

    output = process_images(matcher, images)
    
    save_image(output, 'test_out.tiff')

    plt.imshow(output)
    plt.show()
