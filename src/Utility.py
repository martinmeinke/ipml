'''
Created on Jan 15, 2015

@author: patrik
'''
import os
import pickle
import logging

class TimeManager:
    
    def __init__(self):
        self.start_time = os.times()[4]
        self.elapsed_time = 0
        self.actual_tick = 0
        
    def tick(self):
        self.actual_tick = os.times()[4] - self.elapsed_time-self.start_time
        self.elapsed_time = os.times()[4] - self.start_time
        logging.info("last action: %.2fs; totally elapsed: %.2fs", self.actual_tick, self.elapsed_time)

def LoadPickleFile(path):
    """
    Simply load data from a pickle file
    """
    logging.info("Attempt to load data to file '%s'", path)
    if os.path.exists(path):
        with open(path,'rb') as f:
            data = pickle.load(f)
            logging.info("Data loaded sucessfully")
            return data
    else:
        msg = "File '%s' does not exist" % path
        logging.error(msg)
        raise Exception(msg)
        
def SavePickleFile(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data,f)
    logging.info("Saved data to '%s'", path)
