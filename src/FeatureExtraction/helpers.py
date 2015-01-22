'''
Created on Dec 5, 2014

@author: martin
'''

from PIL import Image
import os
import random
import numpy
import logging
import matplotlib.pyplot as plt
from math import sqrt


def plot_qimage_grayscale(image, title=''):
    """ Plots a quadratic image in grayscale
    the image is provided as a 1 dimensional float ndarray
    """
    #takes a 1d flat representation of the image as input
    assert len(image.shape) == 1
    #edge length of quadratic image
    edgelen = sqrt(image.shape[0])
    img = numpy.reshape(image, (edgelen,edgelen))
    
    plt.imshow(img, cmap = plt.get_cmap("gray"))
    plt.title('Type: {}'.format(title))
    plt.show()
     
def resize_image(image, edgelength):
    #creates quadratic images
    size = (edgelength, edgelength)
    width = image.size[0]
    height = image.size[1]
    
    left = 0
    right = width-1
    top = 0
    bottom = height-1
    
    diff = abs(height-width)
    halfdiff = diff / 2
    
    if width > height: #landscape
        left = halfdiff
        right = right - (diff-halfdiff)
    elif height > width: #portrait
        top = halfdiff
        bottom = bottom - (diff-halfdiff)
        
    box = (left, top, right, bottom)
    image = image.crop(box)

    image.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGBA', size, (255, 255, 255, 0))
    background.paste(
        image,
        ((size[0] - image.size[0]) / 2, (size[1] - image.size[1]) / 2))
    
    return background
        
def resize_images_quadratic(inpath, tgtpath, edgelen=128):
    filenames = os.listdir(inpath)

    for i in filenames:
        im = Image.open(inpath+'/'+i)
        imResize = resize_image(im, edgelen)
        imResize.save(tgtpath + '/' + i, 'JPEG', quality=100)
        logging.info("Saving "+ i)
            
def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.144])
  
def main():
    #resize_images()
    #train, cv, test = create_samples(tstpath)
    pass

if __name__ == '__main__':
    main()
    