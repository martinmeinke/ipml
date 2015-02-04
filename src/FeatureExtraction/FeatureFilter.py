import logging
import numpy as np

class FeatureFilter:
       
    def __init__(self, featureProvider, doglabel, catlabel):
        self._fp = featureProvider
        self._doglabel = doglabel
        self._catlabel = catlabel
        self._meanDiff = 0.0
        self._stdDev = 0.0
        
        self.TrainData = None
        self.TrainLabels = None
        self.ValidationData = None
        self.ValidationLabels = None
        self.TestData = None
        self.TestLabels = None
        
    def initFilter(self):
        trainset = self._fp.TrainData
        labels = np.asarray(self._fp.TrainLabels.ravel())[0]
        
        logging.info("Splitting trainset into dog and cat sets")
        cats = trainset[labels == self._catlabel,:]
        dogs = trainset[labels == self._doglabel,:]
        
        logging.info("Computing means of features for dog and cat classes")
        mean_d = np.mean(dogs, axis=0)
        mean_c = np.mean(cats, axis=0)
        logging.debug("Mean of dogs: %s", str(mean_d))
        logging.debug("Mean of cats: %s", str(mean_c))
        
        logging.info("Computing absolute differences between cat and dog means")
        diffs = np.abs(mean_d - mean_c)
        logging.debug("Diff of means: %s", str(diffs))
        
        logging.info("Computing mean of absolute diffs")
        self._meanDiff = np.mean(diffs)
        logging.info("Mean of diffs is: %s", str(self._meanDiff))

        logging.info("Computing std deviation of absolute diffs")
        self._stdDev = np.std(diffs)
        logging.info("Standard deviation of diffs is: %s", str(self._stdDev))
        self._diffs = np.asarray(diffs)[0]
        
    def applyFilter(self, useStdDev = False):
        threshold = self._meanDiff + self._stdDev if useStdDev else self._meanDiff
        
        logging.info("Filtering data")
        self.TrainData = self._fp.TrainData[:, self._diffs  > threshold]
        self.ValidationData = self._fp.ValidationData[:, self._diffs  > threshold]
        self.TestData = self._fp.TestData[:, self._diffs  > threshold]
        
        self.TrainLabels = self._fp.TrainLabels
        self.ValidationLabels = self._fp.ValidationLabels
        self.TestLabels = self._fp.TestLabels
        
        logging.info("Finished filtering")
        logging.info("Features left after filtering data: %d", self.TrainData.shape[1])
