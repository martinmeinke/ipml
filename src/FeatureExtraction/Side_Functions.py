'''
Created on Jan 15, 2015

@author: patrik
'''
import os
import pickle
import logging

class time_manager:
    
    def __init__(self):
        self.start_time = os.times()[4]
        self.elapsed_time = 0
        self.actual_tick = 0
        
    def tick(self):
        self.actual_tick = os.times()[4] - self.elapsed_time-self.start_time
        self.elapsed_time = os.times()[4] - self.start_time
        logging.info("last action: %.2fs; totally elapsed: %.2fs", self.actual_tick, self.elapsed_time)
