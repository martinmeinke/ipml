'''
Created on Jan 15, 2015

@author: patrik
'''
import numpy as np
from cmath import sqrt
import CPP_Functions
import time

def compute_feature_vector(image, feature_list):
    n = len(feature_list)
    feat_vector = []
    for i in xrange (0,n):
        dist = CPP_Functions.cpp_pic_dist(image, feature_list[i])
        feat_vector.append(dist)
        
    return feat_vector

