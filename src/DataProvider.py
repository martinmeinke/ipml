from os import path

class DataProvider(object):
    '''
    classdocs
    '''
    RAWDATADIR = path.join(path.dirname(path.abspath(__file__)), "../data/")
    CAT_DATAPREFIX = "cat"
    DOG_DATAPREFIX = "dog"


    def __init__(self, rawDataDir = "", catDataPrefix = "", dogDataPrefix = ""):
        self._datadir = rawDataDir or self.RAWDATADIR
        self._catprefix = catDataPrefix or self.CAT_DATAPREFIX
        self._dogprefix = dogDataPrefix or self.DOG_DATAPREFIX
        