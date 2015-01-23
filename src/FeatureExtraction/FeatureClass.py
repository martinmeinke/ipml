'''
Created on Jan 15, 2015

@author: patrik
'''
import os
import numpy as np
import logging
from PIL import Image
import multiprocessing as mp

# import files
import Features
import Vectors
from Utility import TimeManager

def compute_features(path, features):
    img = Image.open(path)
    data = np.asarray(img, np.float32)
    img.close()
    return Vectors.compute_feature_vector(data, features)

class FeatureExtractor(object):
    DISTANCE_THRESHOLD = 100
    NUM_FEATURES = 1000
    MAX_TEXEL_PICS = 5000
    
    
    def __init__(self, distance_threshold = None, num_features = None, max_texel_pics = None):
        # start timer, variables
        self.texel_features = []
        self.feature_border = 5  #!!!cannot be changed!!!
        self.distance_threshold = distance_threshold or self.DISTANCE_THRESHOLD
        self.num_features = num_features or self.NUM_FEATURES
        self.maxTexelPics = max_texel_pics or self.MAX_TEXEL_PICS
        
        self.mytimer = TimeManager()

    def loadState(self, state):
        self.texel_features, self.feature_border, self.distance_threshold, self.num_features, self.maxTexelPics = state


    def initialize(self, trainset):

        logging.info('INTEGRATING IMAGES IN TEXEL FEATURE LIST')

        # restrict the set of images we take the texels from
        # TODO: alternative method: compute how many texels we would have (checking image sizes), then load the texel from file on demand
        #       This approach would take longer but would fit into main memory. We could even implement a buffer of ~2000 images
        #       to speed that method up
        trainset = trainset if len(trainset) < self.maxTexelPics else trainset[:self.maxTexelPics]
        extracted_texels = []
        
        logging.info('cutting images')
    
        for path in trainset:
            img = Image.open(path)
            data = np.array(img)
            img.close()
            extracted_texels += Features.cutimage(data, self.feature_border)
    
        self.mytimer.tick()
        
        logging.info('{0} potential features'.format(len(extracted_texels)))
        logging.info('updating feature list')
        
        self.texel_features = Features.update_feature_list(self.texel_features, extracted_texels, self.distance_threshold, self.num_features)
        
        logging.info('{0} features after update'.format(len(self.texel_features)))
        
        self.mytimer.tick()
        
        logging.info('INTEGRATION DONE')
                
        # self.texel_features = np.asarray(self.texel_features, np.float32)

    def display_data(self):

        logging.info('DISPLAYING TEXELS')

        Features.show_texel_list(self.texel_features)
        self.mytimer.tick()

        logging.info('DISPLAYING DONE')


    def extract(self, trainset):

        n = len(trainset)
        logging.info("COMPUTING %d VECTORS", n)

        nCores = mp.cpu_count()
        pool = mp.Pool(nCores)
        asyncres = []
        for i in xrange(0, n):
            asyncres.append(pool.apply_async(compute_features, [trainset[i], self.texel_features]))

        results = []
        i = 0
        for r in asyncres:
            results.append(r.get())
            i += 1
            if i % 50 == 0:
                logging.info("working... already created a total of %d vectors", i)
                self.mytimer.tick()

        pool.close()
        pool.join()
        self.mytimer.tick()
        logging.info("VECTORS DONE - generated %d vectors", len(results))
        return results

    def saveState(self):
        state = (self.texel_features, self.feature_border, self.distance_threshold, self.num_features, self.maxTexelPics)
        return state
