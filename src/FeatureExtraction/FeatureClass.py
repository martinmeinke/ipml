'''
Created on Jan 15, 2015

@author: patrik
'''
import numpy as np
import logging
from PIL import Image
import multiprocessing as mp
from datetime import datetime, timedelta

# import files
import Features
import Vectors
from Utility import TimeManager

import CPP_Functions
#import Side_Functions

global_feature_images = []
global_labels = []

def compute_distance(texel_feature):
       
    d_dog = 0
    d_cat = 0
    n_dog = 0
    n_cat = 0
    
    diff = 0;
    n_feats = len(texel_feature)
    n_pics = len(global_feature_images)
    
    #print 'Hello'
    
    for j in xrange(0,n_pics):
        
        img = Image.open(global_feature_images[j])
        data = np.asarray(img, np.float32)
        
        #print global_labels[j]
        
        if (global_labels[j] == 1):
            d_dog = d_dog + CPP_Functions.cpp_pic_dist(data, texel_feature)
            n_dog = n_dog +1
        if (global_labels[j] == -1):
            d_cat = d_cat + CPP_Functions.cpp_pic_dist(data, texel_feature)
            n_cat = n_cat +1
            
    d_dog = d_dog / n_dog
    d_cat = d_cat / n_cat
    diff = abs(d_dog - d_cat)
    
    #print 'Hello'
    
    return diff

def compute_features(path, features):
    img = Image.open(path)
    data = np.asarray(img, np.float32)
    return Vectors.compute_feature_vector(data, features)

def extract_texels(trainset, border, distance_threshold, num_features, logger, pipe):
    texels = []
    tm = TimeManager(logger)
    logger.info("sp> Extracting texels in different process")

    for path in trainset:
        img = Image.open(path)
        data = np.array(img)
        texels += Features.cutimage(data, border)

    tm.tick()
    logger.info('sp> {0} potential features'.format(len(texels)))
    logger.info('sp> updating feature list')
    
    features = []
    features = Features.update_feature_list(features, texels, distance_threshold, num_features)

    logger.info('sp> {0} features after update'.format(len(features)))
    tm.tick()

    logger.info("sp> Sending features to pipe")
    pipe.send(features)
    pipe.close()

class FeatureExtractor(object):
    DISTANCE_THRESHOLD = 40
    NUM_FEATURES = 1000
    MAX_TEXEL_PICS = 5000
    
    
    def __init__(self, distance_threshold = None, num_features = None, max_texel_pics = None):
        # start timer, variables
        self.texel_features = []
        self.feature_border = 5  #!!!cannot be changed!!!
        self.distance_threshold = distance_threshold or self.DISTANCE_THRESHOLD
        self.num_features = num_features or self.NUM_FEATURES
        self.maxTexelPics = max_texel_pics or self.MAX_TEXEL_PICS
        
        self.mytimer = TimeManager(logging.getLogger())
        
        self.stats = []

    def loadState(self, state):
        self.texel_features, self.feature_border, self.distance_threshold, self.num_features, self.maxTexelPics = state


    def initialize(self, trainset, labels):

        logging.info('INTEGRATING IMAGES IN TEXEL FEATURE LIST')

        # restrict the set of images we take the texels from
        # TODO: alternative method: compute how many texels we would have (checking image sizes), then load the texel from file on demand
        #       This approach would take longer but would fit into main memory. We could even implement a buffer of ~2000 images
        #       to speed that method up
        trainset = trainset if len(trainset) < self.maxTexelPics else trainset[:self.maxTexelPics]
        labels = labels
        
        global global_feature_images
        global global_labels
            
        global_feature_images = trainset
        global_labels = labels
        
        logging.info('cutting images')
        # Workaround: Python seems to keep internal "free lists" of float values. Extracting the texels
        #             allocates very very much memory without freeing it. Presumably this is because of cached
        #             floats in the float free list. A known workaround is letting a subprocess doing this work
        #             and reading only the features from that process

        parentpipe, childpipe = mp.Pipe()
        p = mp.Process(target=extract_texels, args=(trainset, self.feature_border, self.distance_threshold, self.num_features, logging.getLogger(), childpipe))
        p.start()
        logging.info("Reading features from pipe")
        self.texel_features = parentpipe.recv()
        p.join()
        self.mytimer.tick()
        
        #logging.info("Creating statistics")
        
        
        
        
        logging.info('INTEGRATION DONE')

    def display_data(self):

        logging.info('DISPLAYING TEXELS')

        Features.show_texel_list(self.texel_features)
        self.mytimer.tick()

        logging.info('DISPLAYING DONE')

    def extract(self, trainset, labels):

        comp_stats = 0
        
        if (comp_stats == 1):
            
            n_feats = len(self.texel_features)
            #print 'feats',n_feats
            n_pics = len(trainset)
            count =  0
            d_diff = []
            
            global global_feature_images
            global global_labels
            
            #print '1',global_feature_images
            global_feature_images = trainset
            global_labels = labels
            #print '2',global_feature_images
            
            nCores = mp.cpu_count()
            pool = mp.Pool(nCores)
            results = []
            
            for i in xrange(0,n_feats):
                #print 'Hello'
                #print compute_distance(self.texel_features[i])
                results.append(pool.apply_async(compute_distance, [self.texel_features[i]]))
                #print count
                #count = count + 1
                #self.mytimer.tick()    
            
            for r in results:
                #print r.get()
                d_diff.append(r.get())
            
            pool.close()
            pool.join()
            
            print 'DIFF DONE'
            self.mytimer.tick()
         
            #print d_diff
         
            n_new = len(d_diff)
            mean_dev = 0
            var = 0
            std_dev = 0
             
            mean = np.mean(d_diff)
            
             
            for k in xrange(0,n_new):
                resu = abs(d_diff[k]-mean)
                mean_dev = mean_dev + resu
                var = var + resu*resu/n_new
            mean_dev = mean_dev / n_new
            std_dev = np.sqrt(var)
            
            self.stats.append(mean)
            self.stats.append(mean_dev)
            self.stats.append(var)
            self.stats.append(std_dev)
             
            print 'Mean distance over all {0} texels = {1}'.format(len(self.texel_features),mean)
            print 'Mean deviation over all {0} texels = {1}'.format(len(self.texel_features),mean_dev)
            print 'Variance over all {0} texels = {1}'.format(len(self.texel_features),var)
            print 'Standard deviation over all {0} texels = {1}'.format(len(self.texel_features),std_dev)
            
            #Side_Functions.save_data("../saved/statistics", self.stats)             
                     
            
        elif (comp_stats == 0):
            
            n = len(trainset)
            logging.info("COMPUTING %d VECTORS", n)
    
            nCores = mp.cpu_count()
            pool = mp.Pool(nCores)
            asyncres = []
            for i in xrange(0, n):
                asyncres.append(pool.apply_async(compute_features, [trainset[i], self.texel_features]))
    
            results = []
            workingTimer = TimeManager()
            i = 0
            for r in asyncres:
                results.append(r.get())
                i += 1
                if i % 50 == 0:
                    logging.info("working... already created a total of %d vectors", i)
                    workingTimer.tick()
                    estimatedLeft = float(workingTimer.elapsed_time) / i * (n-i)
                    eta = datetime.now() + timedelta(seconds=int(estimatedLeft))
                    logging.info("           about %.2f seconds left for this vector set", estimatedLeft)
                    logging.info("           estimated end: %s", eta.strftime("%H:%M:%S"))
                    self.mytimer.tick()
    
            pool.close()
            pool.join()
            self.mytimer.tick()
            logging.info("VECTORS DONE - generated %d vectors", len(results))
            return results

    def saveState(self):
        state = (self.texel_features, self.feature_border, self.distance_threshold, self.num_features, self.maxTexelPics)
        return state
