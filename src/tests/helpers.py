'''
Created on Dec 5, 2014

@author: martin
'''

from PIL import Image
import os
import random
from math import floor
    
path = "../../train"
tgtpath = "../../train_resized"
dirs = os.listdir(path)
    
# provide any sequence along with a tuple corresponding to proportions
# TODO: skips a couple of images because of truncating float
def randomly_partition(seq, proportions):
    totalprop = sum(proportions)
    num_images = map(lambda x: (float(x)/totalprop)*len(seq), proportions)
    
    output = []
    
    for partition in num_images:
        output.append([])
        for i in range(int(partition)):
            elementidx = random.randrange(0, len(seq))
            output[-1].append(seq[elementidx])
            del seq[elementidx]
            
    return output        
    
    
def resize_image(image):
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

    size = (128,128)
    image.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGBA', size, (255, 255, 255, 0))
    background.paste(
        image,
        ((size[0] - image.size[0]) / 2, (size[1] - image.size[1]) / 2))
    
    return background

def open_image(path):
    return (Image.open(path).copy(), os.path.split(path)[1])
        
def resize_images():
    for i in dirs:
        im = Image.open(path+'/'+i)
        imResize = resize_image(im)
        imResize.save(tgtpath + '/' + i, 'JPEG', quality=100)
        print "Saving "+ i
            
def main():
    #resize_images()
    images = map(lambda x: open_image(tgtpath+'/'+x), dirs)
    partitions = randomly_partition(images, (6,2,2))
    print partitions

if __name__ == '__main__':
    main()
    