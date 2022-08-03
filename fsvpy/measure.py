import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import numpy as np
import pandas as pd
from skimage import (data, restoration, util)
import skimage
from scipy import ndimage
from skimage.filters import difference_of_gaussians, window, gaussian
import pims
from scipy.optimize import curve_fit
from scipy.special import erf


'''  
    Finds streak attributes

    Input: contours, list of contours, each item in the list contains x & y positions of each contour

    Output: streak_properties, pandas dataframe, lists contour number, area, perimeter, centroid, bbox properties

    Parameters
    ----------

    contour_value: float, threshold to use for identifying contours, if not specified it is set 
                to 5*standard_deviation of image values (preprocessing ensures intensities are 
                gaussian centered at zero)

    frame_number: float, can be used to label frame when batch processing
                '''


def parameters(contours, frame_num = 0):

    property_list = []
    for i, contour in enumerate(contours):

        x = contour[:,1]; y = contour[:,0];  #seperate x & y points

        #check that contour is closed, e.g. it does not interect the image edge, if closed then measure
        if contour[0][0] != contour[-1][0] or contour[0][1] != contour[-1][1]: 
            pass
        else:
            centroid, area, perimeter = contour_shape_specs(x,y)
            corner, width, height = contour_bounding_box(x,y)
            angle = contour_angle(x,y)

            property_list.append([int(i),centroid[0],centroid[1],area, perimeter, corner[0], corner[1], 
                                  width, height, angle,  frame_num])

    properties = pd.DataFrame(data = np.array(property_list), columns = ['streak_id','x','y','area',
                                                                         'perimeter','corner_x', 'corner_y',
                                                                         'bbox_width', 'bbox_height', 'angle','frame'])
    return properties




###################################################################

    '''
    Helper functions to measure contour properties, borrowed heavily from the 'edge' class in ASTiDe

    input: array of x,y points (floats)

    output: contour_shape_specs: centroid, area, perimeter

            contour_bounding_box: bounding box (x,y) corner and width, height

            contour_angle: angle derived from slope fit to all contour points
                            !!this will give nonsense is the streak is not elongated!!!
    '''

def contour_shape_specs(x, y):

    xyxy = (x[:-1] * y[1:] - x[1:] * y[:-1])
    area = 0.5 * np.sum(xyxy)

    perimeter = np.sum(np.sqrt((x[1:] - x[:-1])**2 +
                                   (y[1:] - y[:-1])**2))

    one_sixth_a = 1. / (6. * area)
    x_center = one_sixth_a * np.sum((x[:-1] + x[1:]) * xyxy)
    y_center = one_sixth_a * np.sum((y[:-1] + y[1:]) * xyxy)

    return (x_center, y_center), area, perimeter

def contour_bounding_box(x, y):
    corner = (min(x), min(y))                                                            

    width = max(x)-min(x)
    height = max(y)- min(y)  

    return corner, width, height

def line(x,m,b):
    return m*x + b


def contour_angle(x,y):

        # Fitting a straight line to each edge.
        p0 = [0., 0.]

        p1, s = curve_fit(line, x, y, p0)

        angle = np.arctan(p1[0]) * 180 / np.pi

        return angle




###################################################################################
#final streak fittting functions to get streak width and height, see details below

#fit_shape is the main function
###################################################################################
'''
extract_streaks: extracts piece of original image using streak bbox, should be used for a single frame!
    - if fit fails to converge, streak is rejected
    - if found width or height is more than 20% larger than bounding box, streak is rejected

input: image, numpy array

       df, pandas dataframe of streak properties

       padding: extra distance added to edge of bbox to facilitate fitting

       filepath: string, default is None, 
                  if a filepath is given, images of streaks overlaid with extracted length will be saved


output: df appended with streak width & height
'''
def fit_shape(image, df, padding = 20, pixels_to_average = 2):

    streak_images = []
    widths = []
    heights = []
    slopes = []

    #go through the dataframe row-by-row
    for idx, streak in df.iterrows():

        #extract the bbox of streak with padding
        x0 = int(streak.corner_x - 2*padding); x1 = int(streak.corner_x + streak.bbox_width + 2*padding);
        y0 = int(streak.corner_y - padding); y1 = int(streak.corner_y + streak.bbox_height + padding);
        streak_image = image[y0:y1, x0:x1]

        #rotate iamge
        rotated_streak =  ndimage.rotate(streak_image, streak.angle, reshape=False, mode = 'nearest')

        #find centerline to determine width and height
        xc = int(np.floor(rotated_streak.shape[1] / 2) + 1)
        yc = int(np.floor(rotated_streak.shape[0] / 2) + 1)

        #extract centerline of image for fitting
        width_cut = np.mean(rotated_streak[yc-pixels_to_average : yc+pixels_to_average + 1, :], axis = 0)
        height_cut = np.mean(rotated_streak[:, xc-pixels_to_average : xc+pixels_to_average + 1], axis = 1)      



        #fit for streak width & height
        try:
            w, m = fit_streak_width(width_cut)
        except:
            w = 0
            m = 0
        try:
            h = fit_streak_height(height_cut)
        except:
            h = 0

        #save fits params in array
        widths.append(w); heights.append(h); slopes.append(m);

   #add columns to df in a way that respects indexing and avoids 'Set without copy' warning
    width_series = pd.Series(widths); height_series = pd.Series(heights); slopes_series = pd.Series(slopes);
    width_series.index = df.index; height_series.index = df.index; slopes_series.index = df.index

    df2 = df.copy()

    df2['width'] = width_series
    df2['height'] = height_series
    df2['slope'] = slopes_series

    #do a course filter to remove bad streaks, i.e ones where fit didn't converge or ones which
    #are more than 20% larger than bbox
    df3 = df2[df2.height != 0]
    df4 = df3[df3.width != 0]
    df5 = df4[df4.width < 1.2*df4.bbox_width]
    df6 = df5[df5.height < 1.2*df5.bbox_height]

    return df6

###################################################################
    
#    Helper functions to fit streaks, see details below

###################################################################


'''
convolved erf function for fitting streak width
parameters: 

amp = amplitude
w0: width of bump
L:
s:
m: slope of offset line
'''
def erf_fit(x,amp,w0,L,s,m,b,a):
    erf_convolve = amp*(erf((x-w0)/s/(2**(1/4)))-erf((x-w0-L)/s/(2**(1/4))))/2  
    return erf_convolve*(1-m*x)+b*x+a


'''
gauss function for fitting streak height
parameters: 

amp = amplitude
offset: offset of gaussian center from zero
sigma: width of the bump
m: slope of offset line
'''
def gauss(x,amp,offset,sigma,b,a):
    ans = amp*np.exp(-(x-offset)**2/(2*sigma**2))
    return ans+b*x+a


'''helper function, fits for streak width using erf_convolve
input: centerline: numpy array, bump-like shape assumed

output: w, streak width (width of convolved erf)
'''

def fit_streak_width(centerline):
    #fit for the streak width
    yy = centerline
    xx=np.arange(0,len(yy))

    #normalize 
    yy=(yy-np.min(yy))/(np.max(yy)-np.min(yy))

    #initial guess
    
    #amp,w0,L,s,m,a,b
    amp=1
    w0=20
    L=len(yy)-40
    s=20
    m=(yy[-1]-yy[0])/(xx[-1]-xx[0])
#    a=(yy[-1]-yy[0])/(xx[-1]-xx[0])
    b=0.01##np.mean(yy[0:5])
    a=0.01
    p0=[amp,w0,L,s,m,b,a]

    pred_params, uncert_cov = curve_fit(erf_fit, xx, yy, p0=p0,method='lm')

    # plt.figure()
    # plt.plot(xx,yy,marker='o',c='k',ls='None',ms=3)
    # plt.plot(xx,erf_fit(xx,*pred_params),lw=2,c='r')

    w = pred_params[2]

    return abs(w), pred_params[4]  #there is a degeneracy where sigma can be negative

'''helper function, fits for streak width using erf_convolve
input: centerline: numpy array, bump-like shape assumed

output: h, streak height (width of fit gaussian)
'''

def fit_streak_height(centerline):
             #fit for the streak height
        yy = centerline
        xx = np.arange(0,len(yy))

        #normalize 
        yy=(yy-np.min(yy))/(np.max(yy)-np.min(yy))

        #initial guess
        p0=[1,10,(len(yy)-20),0.01,0.01]

        pred_params, uncert_cov = curve_fit(gauss, xx, yy, p0=p0,method='lm')

        # plt.figure()
        # plt.plot(xx,yy,marker='o',c='k',ls='None',ms=3)
        # plt.plot(xx,gauss(xx,*pred_params),lw=2,c='r')

        h = pred_params[2]

        return abs(h)  #there is a degeneracy where sigma can be negative

