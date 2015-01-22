'''
Created on Dec 5, 2014

@author: martin
'''

from PIL import Image, ImageOps
import os
import random
import numpy
import matplotlib.pyplot as plt
from math import ceil, sqrt, log
import sys
from select import select
import itertools


def plot_qimage_grayscale(image, title=''):
    """ Plots a quadratic image in grayscale
    the image is provided as a 1 dimensional float ndarray
    """
    # takes a 1d flat representation of the image as input
    assert len(image.shape) == 1
    # edge length of quadratic image
    edgelen = sqrt(image.shape[0])
    img = numpy.reshape(image, (edgelen, edgelen))

    plt.figure(99)
    plt.imshow(img, cmap=plt.get_cmap("gray"))
    plt.title('Type: {}'.format(title))
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)
    plt.show()
    plt.figure(1)


def plot_misclassified_images(misclassified_images, test_set, max_images=36, edgelen=128):
    '''
    misclassified_images: list of lists, outer list represents batches
    inner list flags misclassified images within this batch with a 1
    This example: [[0,0,1,0],...] would for example indicate an error
    in the 3rd image of the first set.
    '''
    flattened = numpy.asarray(list(itertools.chain(*misclassified_images)))

    # TODO: nur volle batches!, daher ist flattened kuerzer als train_set_x
    test_set_x = test_set[0]
    set_indices = numpy.where(flattened > 0)[0]

    num_images = min(len(set_indices), max_images)
    edge_len = ceil(sqrt(num_images))

    xpixels = 900
    ypixels = 900

    # get the size in inches
    dpi = 72.
    xinch = xpixels / dpi
    yinch = ypixels / dpi

    plt.figure(99, figsize=(xinch, yinch))

    imshape = (edgelen, edgelen)
    # plot every misclassified image up to max_images
    for i in range(num_images):
        plt.subplot(edge_len, edge_len, i)
        image = test_set_x.container.data[set_indices[i]]

        image = image.reshape(imshape)
        plt.imshow(image, cmap=plt.get_cmap("gray"))

    plt.draw()
    plt.show()
    plt.figure(1)

def plot_kernels(layers, layerids, figids):
    for layer, figid in zip(layerids, figids):
        plt.figure(figid)
        plt.title("Layer: {}".format(layer))

        kernels = layers[layer].fshp[0]

        for i in range(1, kernels + 1):
            plt.subplot(ceil(sqrt(kernels)), ceil(sqrt(kernels)), i)
            kernel = numpy.copy(layers[layer].W.container.data[i - 1][0])
            # normalized = kernel + abs(numpy.min(kernel))
            plt.imshow(kernel, cmap='BrBG', interpolation='none', vmin=-0.5, vmax=0.5)

        plt.draw()
    # switch back to the main figure (TODO: there might be an API call to get
    # the previously active figure)
    plt.figure(1)


def check_break():
    print "Press key to interrupt training after current epoch..."

    rlist, x, y = select([sys.stdin], [], [], 1)

    # key press, leave training
    if rlist:
        return True

    return False


def set_details(s):
    return "#samples: {}, #dogs: {}, #bytes: {}".format(len(s[1]), numpy.sum(s[1]), s[0].nbytes+s[1].nbytes)


def randomly_partition(seq, proportions):
    '''
    provide any sequence along with a tuple corresponding to proportions
    TODO: skips a couple of images because of truncating float
    '''
    totalprop = sum(proportions)
    num_images = map(lambda x: (float(x) / totalprop) * len(seq), proportions)

    output = []

    for partition in num_images:
        output.append([])
        for i in range(int(partition)):  # @UnusedVariable
            elementidx = random.randrange(0, len(seq))
            output[-1].append(seq[elementidx])
            del seq[elementidx]

    return output


def resize_image(image, edgelength, stride=-1):
    '''
    if stride is > 0, not only the center part is cut out
    '''
    # creates quadratic images
    size = (edgelength, edgelength)
    width = image.size[0]
    height = image.size[1]

    ileft = 0
    iright = width - 1
    itop = 0
    ibottom = height - 1

    diff = abs(height - width)

    boxes = []
    newimgs = []

    if stride < 0:
        halfdiff = diff / 2

        if width > height:  # landscape
            left = halfdiff
            right = iright - (diff - halfdiff)
            top = itop
            bottom = ibottom
        elif height > width:  # portrait
            top = halfdiff
            bottom = ibottom - (diff - halfdiff)
            left = ileft
            right = iright

        box = (left, top, right, bottom)
        boxes.append(box)
    else:
        positions = range(0, diff, stride)
        print positions
        if width > height:  # landscape
            for p in positions:
                left = p
                right = iright - (diff - p)
                top = itop
                bottom = ibottom
                boxes.append((left, top, right, bottom))

        elif height > width:  # portrait
            for p in positions:
                top = p
                bottom = ibottom - (diff - p)
                left = ileft
                right = iright
                boxes.append((left, top, right, bottom))

    for box in boxes:
        print box
        imagec = image.copy()
        imagen = imagec.crop(box)
        imagen.thumbnail(size, Image.ANTIALIAS)
        background = Image.new('RGBA', size, (255, 255, 255, 0))
        background.paste(
            imagen,
            ((size[0] - imagen.size[0]) / 2, (size[1] - imagen.size[1]) / 2))

        newimgs.append(background)

    return newimgs


def open_image(path):
    return (Image.open(path).copy(), os.path.split(path)[1])


def resize_images_quadratic(inpath, tgtpath, edgelen=128, stride=-1):
    '''
    if stance is > 0, not only the center part is cut out
    '''
    filenames = os.listdir(inpath)

    if not os.access(tgtpath, os.R_OK):
        os.mkdir(tgtpath)

    for i in filenames:
        im = Image.open(inpath + '/' + i)
        imResize = resize_image(im, edgelen, stride)
        for num in range(len(imResize)):
            sim = imResize[num]
            sim.save(tgtpath + '/' + i.replace(".jpg", "_p{}.jpg".format(num)), 'JPEG', quality=100)
        print "Saving " + i


def isdog(filename):
    if "dog" in filename:
        return 1

    return 0


def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def convert_image_representation(image):
    x = open_image(image)
    return (numpy.array(x[0]), isdog(x[1]))


def create_samples(image_path, partitiondef=(6, 2, 2), partial=100):
    """ creates an arbitrary amount of sample sets
    the number of sets is defined by the number of elements in the
    partitiondef tuple.

    the distribution of the sample images to the sets is done 
    according to the partitiondef tuple.
    (1,1)       = 50%, 50%
    (6,2,2)     = 60%, 20%, 20%
    (0.5,2,2.5) = 10%, 40%, 50%
    """
    # this fails on large images! (not enough ram)
    filenames = os.listdir(image_path)

    if partial != 1:
        upto = partial / 100.0 * len(filenames)
        filenames = numpy.random.permutation(filenames)[0:upto]

    images = map(lambda x: convert_image_representation(
        image_path + '/' + x), filenames)
    # images = map(lambda x: open_image(image_path+'/'+x), filenames)     #[(img, path), (img, path)]
    # images = map(lambda x: (x[0], isdog(x[1])),images)                  #[(img, 1), (img, 0)]
    # images = map(lambda x: (numpy.array(x[0]), x[1]), images)
    # [(arr, 1), (arr, 0)]
    samples = randomly_partition(images, partitiondef)

    # make tuple for each set
    sets = []
    for tset in samples:  # for each set
        sets.append((numpy.asarray(map(lambda x: x[0], tset)), numpy.asarray(
            map(lambda x: x[1], tset))))

    return sets


def lr_decay(lrl, lrt0, decrease_factor, t):
    return lrl / (lrt0 + (log(t*decrease_factor + 2)))-0.012


def plot_lr_decay():
    t = numpy.arange(0, 100, 1)

    ranges = [[0.04],
              [1],
              [0.15]]

    for lrl, lrt0, decfac in list(itertools.product(*ranges)):
        s = map(lambda x: lr_decay(lrl, lrt0, decfac, x), t)
        plt.figure(1)
        plt.plot(t, s, lw=2, label="{}/{}/{}".format(
            lrl, lrt0, decfac))
    plt.legend()
    plt.show()


def flip_images(inpath):
    filenames = os.listdir(inpath)

    for i in filenames:
        im = Image.open(inpath + '/' + i)
        mirror_img = ImageOps.mirror(im)
        mirror_img.save(inpath + '/' + i.replace(".jpg","_mirr.jpg"), 'JPEG', quality=100)


def main():
    # resize_images()
    # train, cv, test = create_samples(tstpath)
    resize_images_quadratic("../../data/test_patches", "../../data/train_images_96_test_enhanced", edgelen=96, stride=5)
    flip_images("../../data/train_images_96_test_enhanced")

    # plot_lr_decay()
    pass

if __name__ == '__main__':
    main()
