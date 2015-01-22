'''
Created on Jan 15, 2015

@author: patrik
'''
# import libraries
import numpy as np
import cv2
import profile

# import files
import FeatureClass

#==== CODE ====

extractor = FeatureClass.feature_extractor(1,0,1,0,1,0)

extractor.extraction_run()
#profile.run("extractor.extraction_run()")