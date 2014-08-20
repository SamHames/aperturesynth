import numpy as np
from skimage import transform
from skimage.feature import match_template


def template_correlate(image,template):
    """ finds the location in pixels of the template in the image"""
    score = match_template(image, template)
    ij = np.unravel_index(np.argmax(score), score.shape)
    x, y = ij[::-1]
    return x,y
    
    
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
        coords.append(np.vstack((rows,cols)).T)
    if pad > 0:    
        return patches,np.vstack(coords)
    else:
        return patches
        
   
class ImageMatcher(object):
    """ Defines an image matching process in terms of a baseline image, and a series of windows
    in that image that define the focal points for that image. """
    def __init__(self, windows, base_image, pad=400):
        # Either the baseline image is loaded, or the fft_patches are specified.
        self.windows = windows
        self.pad = pad
        self.templates = extract_gray_patches(base_image, windows)  
        
    def match(self, image):
        # extract patches and compute phase correlation        
        search_windows,search_coords = extract_gray_patches(image,self.windows,pad=self.pad)
        shifts = np.vstack([template_correlate(window,template) for window,template in zip(search_windows,self.templates)])
        # extract top left of window
        delta = search_coords[::2,[1,0]]-self.windows[::2,[1,0]]
        points1 = self.windows[::2,[1,0]] #+self.windows[1::2,:])[:,[1,0]]/2.0    
        points2 = points1 - shifts - delta #+search_coords[1::2,:])[:,[1,0]]/2.0 - shifts
        match_tform = transform.estimate_transform('similarity',points2,points1)
        return transform.warp(image,match_tform).astype('float32')
        
    def __call__(self,image):
        return self.match(image)
        

    
