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
     
# provide any sequence along with a tuple corresponding to proportions
# TODO: skips a couple of images because of truncating float
def randomly_partition(seq, proportions):
    totalprop = sum(proportions)
    num_images = map(lambda x: (float(x)/totalprop)*len(seq), proportions)
    
    output = []
    
    for partition in num_images:
        output.append([])
        for i in range(int(partition)):  # @UnusedVariable
            elementidx = random.randrange(0, len(seq))
            output[-1].append(seq[elementidx])
            del seq[elementidx]
            
    return output        
    
    
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

def open_image(path):
    return (Image.open(path).copy(), os.path.split(path)[1])
        
def resize_images_quadratic(inpath, tgtpath, edgelen=128):
    filenames = os.listdir(inpath)

    for i in filenames:
        im = Image.open(inpath+'/'+i)
        imResize = resize_image(im, edgelen)
        imResize.save(tgtpath + '/' + i, 'JPEG', quality=100)
        logging.info("Saving "+ i)
            
def isdog(filename):
    if "dog" in filename:
        return 1
    
    return 0

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def convert_image_representation(image):
    x = open_image(image)
    return (numpy.array(x[0]), isdog(x[1]))

def create_samples(image_path, partitiondef = (6,2,2)):
    """ creates an arbitrary amount of sample sets
    the number of sets is defined by the number of elements in the
    partitiondef tuple.
    
    the distribution of the sample images to the sets is done 
    according to the partitiondef tuple.
    (1,1)       = 50%, 50%
    (6,2,2)     = 60%, 20%, 20%
    (0.5,2,2.5) = 10%, 40%, 50%
    """
    #this fails on large images! (not enough ram)
    filenames = os.listdir(image_path)
    #images = map(lambda x: convert_image_representation(image_path+'/'+x), filenames)
    images = map(lambda x: open_image(image_path+'/'+x), filenames)     #[(img, path), (img, path)]
    images = map(lambda x: (x[0], isdog(x[1])),images)                  #[(img, 1), (img, 0)]
    images = map(lambda x: (numpy.array(x[0]), x[1]), images)           #[(arr, 1), (arr, 0)]
    samples = randomly_partition(images, partitiondef)
    
    #make tuple for each set
    sets = []
    for tset in samples: #for each set
        sets.append((numpy.asarray(map(lambda x: x[0], tset)), numpy.asarray(map(lambda x: x[1], tset))))
                
    return sets
    
def main():
    #resize_images()
    #train, cv, test = create_samples(tstpath)
    pass

if __name__ == '__main__':
    main()
    