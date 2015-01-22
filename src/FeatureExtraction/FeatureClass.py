'''
Created on Jan 15, 2015

@author: patrik
'''
import numpy as np
import cv2

# import files
import helpers
import Features
import Vectors
import Side_Functions
#from Pack1.Main2 import mytimer, feature_images, texel_features,\
#    filepath_texel_features

class feature_extractor(object):
    
    def __init__(self, integrate_images_bit = 1, display_features_bit = 0, compute_vectors_bit = 1, load_data_bit = 0, store_data_bit = 1,load_test_bit=1):
        #start timer, variables
        self.texel_features = []
        self.dog_vectors = []
        self.cat_vectors = []
        self.feature_images = []

        self.distance_threshold = 100
        self.feature_border = 5 #!!!cannot be changed!!!
        self.partition1 = (10,0,0)
        self.max_num_of_texels = 100
        self.max_num_of_vectors = 100
        self.filepath_texel_features = "save/texel_features"
        self.filepath_dog_vectors = "save/dog_vectors"
        self.filepath_cat_vectors = "save/cat_vectors"
        
        self.load_data_bit = load_data_bit
        self.store_data_bit = store_data_bit
        self.integrate_images_bit = integrate_images_bit
        self.display_features_bit = display_features_bit
        self.compute_vectors_bit = compute_vectors_bit
        self.load_test_bit = load_test_bit
        
        self.mytimer = Side_Functions.time_manager()
        

    def partition_data(self):
        #FEATURE EXTRACTION
        print 'PARTIONING DATA FOR EXTRACTION OF FEATURES'

        set3 = helpers.create_samples("../../data", self.partition1)
        self.feature_images = set3[0]
        self.mytimer.tick()

        print 'PARTITIONING DONE'

    def load_data(self):

        if(self.load_data_bit):
            print 'LOADING TEXEL FEATURES'
    
            self.texel_features = Side_Functions.load_file(self.filepath_texel_features)
            self.mytimer.tick()
    
            print 'LOADING DONE'

    def integrate_data(self):

        if(self.integrate_images_bit):
            
            print 'INTEGRATING IMAGES IN TEXEL FEATURE LIST'

            n = len(self.feature_images[0])
            extracted_texels = []
            
            print 'cutting images'
        
            for i in xrange(0,n):
                
                extracted_texels = Features.cutimage(self.feature_images[0][i], self.feature_border, extracted_texels)
        
            self.mytimer.tick()
            
            print '{0} potential features'.format(len(extracted_texels))
            print 'updating feature list'
            
            self.texel_features = Features.update_feature_list(self.texel_features, extracted_texels, self.distance_threshold, self.max_num_of_texels)
            
            print '{0} features after update'.format(len(self.texel_features))
            
            self.mytimer.tick()
            
            print 'INTEGRATION DONE'
                    
            #self.texel_features = np.asarray(self.texel_features, np.float32)

    def display_data(self):

        if(self.display_features_bit):
            print 'DISPLAYING TEXELS'
    
            Features.show_texel_list(self.texel_features)
            self.mytimer.tick()
    
            print 'DISPLAYING DONE'

    def compute_vectors(self):

        if(self.compute_vectors_bit):
            print 'COMPUTING VECTORS'
    
            n = len(self.feature_images[0])

            for i in range(0,n):
                if (self.feature_images[1][i] == 1): #-->1 = dog
                    if(len(self.dog_vectors) < self.max_num_of_vectors):
                        img = np.asarray(self.feature_images[0][i], np.float32)
                        vect = Vectors.compute_feature_vector(img, self.texel_features)
                        self.dog_vectors.append(vect)
                        print 'Vector {} integrated'.format(i)
                        self.mytimer.tick()
                else:
                    if(len(self.cat_vectors) < self.max_num_of_vectors):
                        img = np.asarray(self.feature_images[0][i], np.float32)
                        vect = Vectors.compute_feature_vector(img, self.texel_features)
                        self.cat_vectors.append(vect)
                        print 'Vector {} integrated'.format(i)
                        self.mytimer.tick()            
    
        print 'VECTORS DONE - generated {0} dog vectors and {1} cat vectors'.format(len(self.dog_vectors),len(self.cat_vectors))

    def store_data(self):

        if(self.store_data_bit):
            print 'STORING TEXEL FEATURES'
    
            Side_Functions.save_data(self.filepath_texel_features, self.texel_features)
            self.mytimer.tick()
    
            print 'STORING DOG VECTORS'
            Side_Functions.save_data(self.filepath_dog_vectors, self.dog_vectors)
            self.mytimer.tick()
    
            print 'STORING CAT VECTORS'
            Side_Functions.save_data(self.filepath_cat_vectors, self.cat_vectors)
            self.mytimer.tick()
    
        print 'STORING DONE'

    def load_test(self):

        if(self.load_test_bit):
            print 'LOAD TEST FOR STORED DATA'
            feats = Side_Functions.load_file(self.filepath_texel_features)
            #print 'Features: {}'.format(feats)
            print "feature 2", feats[2]
            print "feature 12", feats[12]
            print "feature 34", feats[19]
            dogs = Side_Functions.load_file(self.filepath_dog_vectors)
            #print 'Dogs: {}'.format(dogs)
            print "dogs 2", dogs[2]
            print "dogs 4", dogs[4]
            print "dogs 7", dogs[7]
            cats = Side_Functions.load_file(self.filepath_cat_vectors)
            #print 'Cats: {}'.format(cats)
            print "cats 1", cats[1]
            print "cats 4", cats[4]
            print "cats 6", cats[6]
    
    def extraction_run(self):
        
        self.partition_data()
        self.load_data()
        self.integrate_data()
        self.display_data()
        self.compute_vectors()
        self.store_data()
        self.load_test()       

        print 'END'