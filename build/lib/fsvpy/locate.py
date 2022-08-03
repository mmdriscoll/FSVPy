import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import skimage
from scipy import ndimage
from skimage.filters import difference_of_gaussians, gaussian
import pims



'''

Performs bandpass filtering.  Gaussian blurs result to smooth pixel noise to help the contour finding.


Parameters
----------
image: numpy array

low_sigma : int, low frequency cutoff for bandpass filter

blur_sigma: int, radius for gaussian blur (done to aid contour finding after filtering)

Returns
-------
processed_image: numpy array 

'''

def preprocess_image(image, low_sigma = 1, blur_sigma = 3):

    image_result = difference_of_gaussians(image, low_sigma = low_sigma)  #bandpass filter

    image_result = gaussian(image_result, sigma = blur_sigma)  #blur to aid contour finding
    
    return image_result


    '''
    
    Finds contours in the image after preprocessing.

    Input: image (array-like)

    Output: list of contours, each item in the list contains x & y positions of each contour

    Parameters
    ----------
    image: numpy array

    contour_value: float, threshold to use for identifying contours, if not specified it is set 
                to 5*standard_deviation of image values (preprocessing ensures intensities are 
                gaussian centered at zero)


    '''
def locate_streaks(image, contour_value = None):

    processed_image = preprocess_image(image)

    if contour_value == None:
        contour_value = 5*np.std(processed_image)
  
    contours =  skimage.measure.find_contours(processed_image, contour_value) 

    return contours

   