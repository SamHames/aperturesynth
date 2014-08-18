#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic Aperture: A tool for registering and fusing a series of handheld images.

"""

import numpy
import matplotlib.pyplot as plt 
from PIL import Image
from skimage import transform
from skimage.feature import match_template
import multiprocessing as mp
import sys


# TODO: make this sub pixel    
def template_correlate(image,template):
    """ finds the location in pixels of the template in the image"""
    score = match_template(image, template)
    ij = numpy.unravel_index(numpy.argmax(score), score.shape)
    x, y = ij[::-1]
    return x,y
    

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
    
    
def extract_gray_patches(image,windows,pad=0):
    """ return a list of image patches defined by the windows as corners of patches """
    patches = []
    coords = []
    max_rows,max_cols = image.shape[:2]
    n_windows = int(windows.shape[0]/2)
    for i in range(n_windows):
        rows = windows[i*2:i*2+2,0] + [-pad,pad] 
        cols = windows[i*2:i*2+2,1] + [-pad,pad]
        rows[rows<0] = 0; rows[rows>max_rows] = max_rows
        cols[cols<0] = 0; cols[cols>max_cols] = max_cols
        patches.append(image[rows[0]:rows[1],cols[0]:cols[1],1])
        coords.append(numpy.vstack((rows,cols)).T)
    if pad > 0:    
        return patches,numpy.vstack(coords)
    else:
        return patches


def trim_shift(image, coords):
    """ trims the output so that pixels not fully defined are removed """
    xshift = coords[:,0]; yshift = coords[:,1]
    left = max(xshift) if max(xshift) > 0 else 0
    right = min(xshift) if min(xshift) < 0 else -1
    top = max(yshift) if max(yshift) > 0 else 0
    bottom = min(yshift) if min(yshift) < 0 else -1    
    return image[left:right,top:bottom,:]
    
    
def determine_output_size(transforms):
    """ return the size and offset for the canvas to hold all of the tranformed images """
    return None
    

def save_image(image,filename):
    """ saves the image to the given filename, ensuring reasonable range"""
     # clamp out of range colours
    imageIn = image.copy()
    imageIn[imageIn > 1] = 1
    imageIn[imageIn < 0] = 0
    imageIn *= 255
    im = Image.fromarray(imageIn.astype(numpy.uint8))
    im.save(filename)
   

# One way to improve the matching process might be to iterate: compute a transform,
# then repeat with the transformed image to correct for inaccuracy due to rotation.
    
class ImageMatcher(object):
    """ Defines an image matching process in terms of a baseline image, and a series of windows
    in that image that define the focal points for that image. """
    def __init__(self,windows, pad=400, baseline=None, templates=None):
        # Either the baseline image is loaded, or the fft_patches are specified.
        self.windows = windows
        self.pad = pad
        if baseline is not None:
            baseline = numpy.asarray(Image.open(baseline),dtype='float32')/255.  
            self.templates = extract_gray_patches(baseline,windows)
        else:
            self.templates = templates   
            
    def match(self, image):
        image = numpy.asfarray(Image.open(image),dtype='float32')/255. 
        # extract patches and compute phase correlation        
        search_windows,search_coords = extract_gray_patches(image,self.windows,pad=self.pad)
        shifts = numpy.vstack([template_correlate(window,template) for window,template in zip(search_windows,self.templates)])
        # extract top left of window
        delta = search_coords[::2,[1,0]]-self.windows[::2,[1,0]]
        points1 = self.windows[::2,[1,0]] #+self.windows[1::2,:])[:,[1,0]]/2.0    
        points2 = points1 - shifts - delta #+search_coords[1::2,:])[:,[1,0]]/2.0 - shifts
        match_tform = transform.estimate_transform('similarity',points2,points1)
        return transform.warp(image,match_tform).astype('float32')
        
    def __call__(self,image):
        return self.match(image)
        
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

# TODO Try using guided filtering as part of the fusion process.

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
    # TODO: add some message passing from the workers when an image is done for progress 
    # reporting later.
    # may need to add back the accumulator process for the gui?
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
    
def smooth_edges(mean_image,base_image,levels):
    """ smooths the mean_image, using the base image to construct an edge map"""
    edge_mask = 1 - numpy.abs(mean_image-base_image)
    mean_pyr = transform.pyramid_laplacian(mean_image,max_layers=levels)
    edge_pyr = transform.pyramid_gaussian(edge_mask,max_layers=levels)
    masked = []
    for mean,mask in zip(mean_pyr,edge_pyr):
        masked.append(mean*mask)
    
    for index in range(len(masked)-1,-1,-1):
        next_layer = masked[index]
        
        
        
    
def main(list_images,out_file):
    # load the first image, get the windows
    focus_windows = get_windows(list_images[0])
    matcher = ImageMatcher(focus_windows,baseline=list_images[0])
    output = process_images(matcher,list_images)
    save_image(output,out_file)
    # accumulate the images, using several processors
    # save the output file
          
     
if __name__ == "__main__":
    images = sys.argv[1:]
    out_file = images[-1]
    main(images[:-1],out_file)
    
    
