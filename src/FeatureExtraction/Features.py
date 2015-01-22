'''
Created on Jan 15, 2015

@author: patrik
'''

import numpy as np
import CPP_Functions
import logging
from matplotlib import pyplot as plt
from math import floor


def cutimage(image,edgesize):
    
    h,w,d = image.shape
    ddelta = edgesize
    rrow = h / ddelta
    ccol = w / ddelta
    features = []

    for r in xrange(0,rrow):
        for c in xrange(0,ccol):
            texel = image[r*ddelta:(r+1)*ddelta,c*ddelta:(c+1)*ddelta]
            features.append(texel)
            
    return features

def dist_from_texel_set(texel_set,texel):
    
    n = len(texel_set)
    d_min = 255*np.sqrt(3)
    d_max = 0
    
    for i in xrange(0,n):
        #d = texel_dist(texel, texel_set[i])
        d = CPP_Functions.cpp_tex_dist(texel, texel_set[i])
        if d > d_max:
            d_max = d
        if d < d_min:
            d_min = d
        
    return d_max,d_min 

def update_feature_list(feature_list, texel_list, dist_threshold, max_num_of_feat):
    
    n = len(texel_list)
    c_added = 0
    c_discarted = 0
    
    while((n > 0) and (len(feature_list) < max_num_of_feat)):
        i = n*np.random.rand()
        #print i
        i = np.int(i)
        #print i
        texel = np.asarray(texel_list[i], np.float32)
        #d = dist_from_texel_set(feature_list, texel)                #testzeile
        #print 'd',d                                                 #testzeile
        d_max,d_min = dist_from_texel_set(feature_list, texel)
        #print 'd_max,d_min',d_max,d_min
        if d_min > dist_threshold:
            feature_list.append(texel)
            c_added = c_added + 1
        else:
            c_discarted =  c_discarted +1
        texel_list.pop(i)
        n = len(texel_list)
    
    logging.info("{0} features added // {1} features discarted".format(c_added,c_discarted))
    return feature_list

def show_texel_list(texel_list):
        
    n = len(texel_list)
    
    ro = floor(np.sqrt(n))+1
    co = floor(np.sqrt(n))+1
        
    for i in xrange(0,n):
        plt.subplot(ro,co,i+1),plt.imshow(texel_list[i]) #cmap=plt.get_cmap('jet')
        #plt.jet()
        plt.xticks([]),plt.yticks([])
    
    plt.show()