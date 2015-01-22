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
import Side_Functions
from Utility import LoadPickleFile, SavePickleFile

def compute_features(path, features):
    img = Image.open(path)
    data = np.asarray(img, np.float32)
    img.close()
    return Vectors.compute_feature_vector(data, features)

class feature_extractor(object):
    DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../features/")
    CAT_FEATURES_FILE = os.path.join(DATADIR, "cat_vectors")
    DOG_FEATURES_FILE = os.path.join(DATADIR, "dog_vectors")
    EXTRACTOR_DATA_FILE = os.path.join(DATADIR, "texel_features")
    
    def __init__(self, dataprovider, integrate_images_bit=1, display_features_bit=0, compute_vectors_bit=1, load_data_bit=0, store_data_bit=1, load_test_bit=1):
        # start timer, variables
        self._dataprovider = dataprovider
        self.texel_features = []
        self.dog_vectors = []
        self.cat_vectors = []
        self.feature_images = []

        self.distance_threshold = 100
        self.feature_border = 5  #!!!cannot be changed!!!
        self.partition1 = (10, 0, 0)
        self.max_num_of_texels = 100
        self.max_num_of_vectors = 100
        
        self.load_data_bit = load_data_bit
        self.store_data_bit = store_data_bit
        self.integrate_images_bit = integrate_images_bit
        self.display_features_bit = display_features_bit
        self.compute_vectors_bit = compute_vectors_bit
        self.load_test_bit = load_test_bit
        
        self.mytimer = Side_Functions.time_manager()

    def load_data(self):

        if(self.load_data_bit):
            logging.info('LOADING TEXEL FEATURES')
    
            self.texel_features = Side_Functions.load_file(self.filepath_texel_features)
            self.mytimer.tick()
    
            logging.info('LOADING DONE')

    def integrate_data(self):

        if(self.integrate_images_bit):
            
            logging.info('INTEGRATING IMAGES IN TEXEL FEATURE LIST')

            trainset = self._dataprovider.TrainData
            extracted_texels = []
            
            logging.info('cutting images')
        
            for img in trainset:
                extracted_texels += Features.cutimage(self.read_img(img), self.feature_border)
        
            self.mytimer.tick()
            
            logging.info('{0} potential features'.format(len(extracted_texels)))
            logging.info('updating feature list')
            
            self.texel_features = Features.update_feature_list(self.texel_features, extracted_texels, self.distance_threshold, self.max_num_of_texels)
            
            logging.info('{0} features after update'.format(len(self.texel_features)))
            
            self.mytimer.tick()
            
            logging.info('INTEGRATION DONE')
                    
            # self.texel_features = np.asarray(self.texel_features, np.float32)

    def display_data(self):

        if(self.display_features_bit):
            logging.info('DISPLAYING TEXELS')
    
            Features.show_texel_list(self.texel_features)
            self.mytimer.tick()
    
            logging.info('DISPLAYING DONE')

    def read_img(self, path):
        img = Image.open(path)
        data = np.array(img)
        img.close()
        return data

    def compute_vectors(self):
        if(not self.compute_vectors_bit):
            return

        logging.info('COMPUTING VECTORS')

        trainset = self._dataprovider.TrainData
        trainlabels = self._dataprovider.TrainLabels
        n = len(trainset)

        nCores = mp.cpu_count()
        pool = mp.Pool(nCores)
        results = []
        numres = (0,0)
        for i in xrange(0, n):
            resultflag = 0 if trainlabels[i] == self._dataprovider.DogLabel else 1
            if numres[resultflag] > self.max_num_of_vectors:
                continue
            results.append((pool.apply_async(compute_features, [trainset[i], self.texel_features]), resultflag))

        for r in results:
            affectedVec = self.dog_vectors if r[1] == 0 else self.cat_vectors
            affectedVec.append(r[0].get())

        pool.close()
        pool.join()
        self.mytimer.tick()
        logging.info('VECTORS DONE - generated {0} dog vectors and {1} cat vectors'.format(len(self.dog_vectors), len(self.cat_vectors)))

    def store_data(self, catPath = "", dogPath = "", featuresPath =""):
        catPath = catPath or self.CAT_FEATURES_FILE
        dogPath = dogPath or self.DOG_FEATURES_FILE
        featuresPath = featuresPath or self.EXTRACTOR_DATA_FILE

        if(self.store_data_bit):
            logging.info('STORING TEXEL FEATURES')
    
            SavePickleFile(featuresPath, self.texel_features)
            self.mytimer.tick()
    
            logging.info('STORING DOG VECTORS')
            SavePickleFile(dogPath, self.dog_vectors)
            self.mytimer.tick()
    
            logging.info('STORING CAT VECTORS')
            SavePickleFile(catPath, self.cat_vectors)
            self.mytimer.tick()

        logging.info('STORING DONE')

    def load_test(self):

        if(self.load_test_bit):
            logging.info('LOAD TEST FOR STORED DATA')
            feats = Side_Functions.load_file(self.filepath_texel_features)
            # print 'Features: {}'.format(feats)
            logging.info("feature 2", feats[2])
            logging.info("feature 12", feats[12])
            logging.info("feature 34", feats[19])
            dogs = Side_Functions.load_file(self.filepath_dog_vectors)
            # print 'Dogs: {}'.format(dogs)
            logging.info("dogs 2", dogs[2])
            logging.info("dogs 4", dogs[4])
            logging.info("dogs 7", dogs[7])
            cats = Side_Functions.load_file(self.filepath_cat_vectors)
            # print 'Cats: {}'.format(cats)
            logging.info("cats 1", cats[1])
            logging.info("cats 4", cats[4])
            logging.info("cats 6", cats[6])
    
    def extraction_run(self):
        
        self.load_data()
        self.integrate_data()
        self.display_data()
        self.compute_vectors()
        self.store_data()
        self.load_test()       

        logging.info('END')
