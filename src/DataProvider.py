
import os
import re
import random
import logging

from Utility import LoadPickleFile, SavePickleFile

class DataProvider(object):
    '''
    Reads the files from a directory and generates data and label sets for training, validation and testing data.
    The data sets contain the absolute file path, the label sets contain CAT_LABEL and DOG_LABEL, corresponding
    to whether the data is a dog or cat file.
    '''
    RAWDATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")
    CAT_DATAPREFIX = "cat"
    DOG_DATAPREFIX = "dog"
    IMG_EXT = "jpg"
    SAVEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../saved/data_segmentation")
    DEFAULT_SEGMENTATION = (70, 15, 15)
    MAXFILES = -1
    
    CAT_LABEL = -1
    DOG_LABEL = 1

    def __init__(self, catLabel = None, dogLabel = None, rawDataDir = "", catDataPrefix = "", dogDataPrefix = ""):
        self.CatLabel = catLabel if catLabel != None else self.CAT_LABEL # don't use "or" syntax to allow 0 as value
        self.DogLabel = dogLabel if dogLabel != None else self.DOG_LABEL # don't use "or" syntax to allow 0 as value
        self._datadir = rawDataDir or self.RAWDATADIR
        self._catprefix = catDataPrefix or self.CAT_DATAPREFIX
        self._dogprefix = dogDataPrefix or self.DOG_DATAPREFIX
        self.TrainData = None
        self.TrainLabels = None
        self.ValidationData = None
        self.ValidationLabels = None
        self.TestData = None
        self.TestLabels = None

    def initialize(self, segmentation = None, maxfiles = 0):
        segmentation = segmentation or self.DEFAULT_SEGMENTATION
        maxfiles = maxfiles or self.MAXFILES
        # normalize segmentation
        segmentation = map(lambda x : float(x) / sum(segmentation), segmentation)

        label = lambda x : self.CatLabel if x.startswith(self.CAT_DATAPREFIX) else self.DogLabel # can only be DOG_PREFIX otherwise because of file filter
        # list available files
        filematcher = re.compile("^({}|{})\.[0-9]+\.{}$".format(self._catprefix, self._dogprefix, self.IMG_EXT), re.IGNORECASE)
        logging.info("Creating file list")
        labeledfiles = [(os.path.join(self.RAWDATADIR, f), label(f)) for f in os.listdir(self.RAWDATADIR) if filematcher.match(f)]
        logging.info("Shuffling file list")
        # we shuffle twice. why? because python docs say that for bigger sequences only some permutations can be created
        # because of the limits of the pseudo random number generator. doing it twice we at least increase the chances for more randomness
        random.shuffle(labeledfiles)
        random.shuffle(labeledfiles)
        
        if maxfiles > 0 and len(labeledfiles) > maxfiles:
            del labeledfiles[maxfiles:]

        nData = len(labeledfiles)
        nTrain = int(segmentation[0] * nData)
        nValidation = int(segmentation[1] * nData)
        nTest = nData - nTrain - nValidation
        logging.info("{} files in total. Using {} for training, {} for validation, and {} for testing".format(nData, nTrain, nValidation, nTest))
        self.TrainData =   [x[0] for x in labeledfiles[0:nTrain]]
        self.TrainLabels = [x[1] for x in labeledfiles[0:nTrain]]
        self.ValidationData =   [x[0] for x in labeledfiles[nTrain:nTrain+nValidation]]
        self.ValidationLabels = [x[1] for x in labeledfiles[nTrain:nTrain+nValidation]]
        self.TestData =   [x[0] for x in labeledfiles[nTrain+nValidation:]]
        self.TestLabels = [x[1] for x in labeledfiles[nTrain+nValidation:]]
        logging.info("Segmentation finished.")

    def saveToFile(self, path = ""):
        path = path or self.SAVEPATH
        logging.info("Saving data segmentation")
        SavePickleFile(self.SAVEPATH, (self.TrainData, self.TrainLabels, self.ValidationData, self.ValidationLabels, self.TestData, self.TestLabels))

    def loadFromFile(self, path = ""):
        path = path or self.SAVEPATH
        logging.info("Loading data segmentation")
        self.TrainData, self.TrainLabels, self.ValidationData, self.ValidationLabels, self.TestData, self.TestLabels = LoadPickleFile(self.SAVEPATH)
