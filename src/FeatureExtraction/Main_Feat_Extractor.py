'''
Created on Jan 15, 2015

@author: patrik
'''
# import libraries

# import files
import FeatureClass
import logging
import sys

#==== CODE ====

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s") # print logging to stdout by default
    extractor = FeatureClass.feature_extractor(1,0,1,0,1,0)
    extractor.extraction_run()
    #profile.run("extractor.extraction_run()")

if __name__ == "__main__":
    main()