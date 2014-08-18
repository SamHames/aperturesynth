import multiprocessing as mp


def transform_worker(matcher,image_queue,transformed_queue):
    """ processes the images from the queue, one at a time """
    images = 0    
    for image in iter(image_queue.get,'STOP'):
        if images == 0:
            acc = matcher(image)
            images += 1
        else:
            acc += matcher(image)
    transformed_queue.put(acc)


def process_images(matcher,image_list,filename=None,n_workers=0):
    """ Processes the images in image list, using matcher, in a parallel fashion."""
    image_queue = mp.Queue()
    accumulate_queue = mp.Queue()
    
    for image in image_list[1:]: 
        image_queue.put(image) 
        
    if n_workers == 0:
        n_workers = int(mp.cpu_count()/2) #Hack to account for hyperthreading, and also not consume all available resources
    
    processes = []
    # start a series of workers for each process
    for i in range(n_workers):
        p = mp.Process(target=transform_worker,args=(matcher,image_queue,accumulate_queue))
        p.start()
        processes.append(p)
        image_queue.put('STOP')
    
    procs_done = 0
    acc = numpy.asarray(Image.open(image_list[0]),dtype='float32')/255.
    for accumulated in iter(accumulate_queue.get,'DUMMY'):
        acc += accumulated
        procs_done += 1
        if procs_done == n_workers:
            break      
    # finish everything off
    print("Terminating processes")
    for p in processes: 
        p.join()
    acc /= len(image_list)
    return acc        
