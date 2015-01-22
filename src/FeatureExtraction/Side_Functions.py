'''
Created on Jan 15, 2015

@author: patrik
'''
import os.path
import pickle
import time
import logging

class time_manager:
    
    def __init__(self):
        self.start_time = time.clock()
        self.elapsed_time = 0
        self.actual_tick = 0
        
    def tick(self):
        self.actual_tick = time.clock() - self.elapsed_time-self.start_time
        self.elapsed_time = time.clock() - self.start_time
        logging.info("last action: {}; totally elapsed: {}".format(self.actual_tick, self.elapsed_time))
        
def load_file(file_path):
    
    if os.path.exists(file_path):
        logging.info('File found - Loading feature list')
        with open(file_path,'rb') as f:
            loaded_data = pickle.load(f)
            return loaded_data
    else:
        logging.info("File not found in specified directory")
    
def save_data(file_path, data_to_store):
    
    with open(file_path, 'wb') as f:
        pickle.dump(data_to_store,f)
    logging.info('Data Saved')