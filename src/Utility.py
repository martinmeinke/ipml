'''
Created on Jan 15, 2015

@author: patrik
'''
import os
import pickle
import logging
import gzip

class TimeManager:
    
    def __init__(self, logger = None):
        self._logger = logger
        self.start_time = os.times()[4]
        self.elapsed_time = 0
        self.actual_tick = 0
        
    def tick(self):
        self.actual_tick = os.times()[4] - self.elapsed_time-self.start_time
        self.elapsed_time = os.times()[4] - self.start_time
        if self._logger:
            self._logger.info("last action: %.2fs; totally elapsed: %.2fs", self.actual_tick, self.elapsed_time)

def LoadPickleFile(path):
    """
    Simply load data from a pickle file
    """
    logging.info("Attempt to load data to file '%s'", path)
    if os.path.exists(path):
        f = gzip.open(path,'rb')
        data = pickle.load(f)
        f.close()
        logging.info("Data loaded sucessfully")
        return data
    else:
        msg = "File '%s' does not exist" % path
        logging.error(msg)
        raise Exception(msg)
        
def SavePickleFile(path, data):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    f = gzip.open(path, 'wb')
    pickle.dump(data, f)
    f.close()
    logging.info("Saved data to '%s'", path)

def LimitDataSets(datasets, sumlimit):
    logging.info("Limiting data to a total of %d", sumlimit)
    segmentation = map(len, datasets)
    logging.info("Set lengths before: %s", str(segmentation))
    segmentation = map(lambda x : float(x) / sum(segmentation), segmentation)
    limited = []
    for i in range(0, len(datasets)):
        limit = int(sumlimit * segmentation[i])
        limited.append(datasets[i][:limit] if limit < len(datasets[i]) else datasets[i])
    logging.info("Set lengths after: %s", str(map(len, limited)))
    return limited

