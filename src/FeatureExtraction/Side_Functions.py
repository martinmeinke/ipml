'''
Created on Jan 15, 2015

@author: patrik
'''
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mathtext import DELTA
from cmath import sqrt
from bzrlib.upgrade import Convert
from math import floor
from cv2 import waitKey
from matplotlib.pyplot import jet
import os.path
import pickle
import time
import CPP_Functions

class time_manager:
    
    def __init__(self):
        self.start_time = time.clock()
        self.elapsed_time = 0
        self.actual_tick = 0
        
    def tick(self):
        self.actual_tick = time.clock() - self.elapsed_time-self.start_time
        self.elapsed_time = time.clock() - self.start_time
        print 'last action: ', self.actual_tick, '; totally elapsed: ', self.elapsed_time
        
def load_file(file_path):
    
    if os.path.exists(file_path):
        print 'File found - Loading feature list'
        with open(file_path,'rb') as f:
            loaded_data = pickle.load(f)
            return loaded_data
    else:
        print "File not found in specified directory"
    
def save_data(file_path, data_to_store):
    
    with open(file_path, 'wb') as f:
        pickle.dump(data_to_store,f)
    print 'Data Saved'