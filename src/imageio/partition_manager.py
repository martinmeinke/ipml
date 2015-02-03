'''
Created on Dec 11, 2014

@author: martin
'''

import logging
from math import ceil

logger = logging.getLogger(__name__)


class PartitionManager(object):
    '''
    classdocs
    '''
    MAX_BATCHES_PER_PARTITION = 50
    num_partitions = []
    current_partition = []
    
    def __init__(self, n_batches, config_methods):
        
        '''
        Constructor
        '''
        
        logger.info("Initializing partition manager")

        self.num_batches = n_batches
        self.config_methods = config_methods
        
        for b in self.num_batches:
            n = ceil(b / self.MAX_BATCHES_PER_PARTITION)
            logger.info("Number of partitions: {}".format(n))
            self.num_partitions.append(n)
            
        # swap in initial partition?
        for b in self.num_batches:
            self.current_partition.append(0)
        
    def swap_in_partition(self, setid, minibatch_index):
        logger.debug("Checking if swap is required for minibatch: {}".format(minibatch_index))
        tgt_partition = minibatch_index / self.MAX_BATCHES_PER_PARTITION
        if self.current_partition[setid] != tgt_partition:
            logger.info("Swapping partition {} to shared memory".format(tgt_partition))
            self.config_methods[setid](tgt_partition)
            self.current_partition[setid] = tgt_partition
            
        return minibatch_index % self.MAX_BATCHES_PER_PARTITION