import logging
import numpy as np

class FeatureFilter:
       
    def __init__(self, featureProvider, doglabel, catlabel):
        self._fp = featureProvider
        self._doglabel = doglabel
        self._catlabel = catlabel
        self._filter = None
        
        self.TrainData = None
        self.TrainLabels = None
        self.ValidationData = None
        self.ValidationLabels = None
        self.TestData = None
        self.TestLabels = None
        
    def initFilter(self, filterName = 'mean', **kwargs):
        trainset = self._fp.TrainData
        labels = np.asarray(self._fp.TrainLabels.ravel())[0]
        
        if filterName == 'mean':
            logging.info("Using filter based of mean feature differences between classes")
            self.initMeanFilter(trainset, labels, **kwargs)
        elif filterName == 'variance':
            logging.info("Using low variance feature filter")
            self.initVarianceFilter(trainset, labels, **kwargs)
        else:
            msg = "Unknown filter with name '%s'" % filterName
            logging.error(msg)
            raise Exception(msg)

    def initVarianceFilter(self, trainset, labels):
        logging.info("Computing variances of features")
        variances = np.var(trainset, axis=0)
        logging.debug("Variances shape: %s", str(variances.shape))
        logging.debug("Variances of features are: %s", str(variances))
        logging.debug("Mean of variances: %s", str(np.mean(variances)))
        variances = np.asarray(variances)[0]
        self._filter = variances > np.mean(variances)
        
    def initMeanFilter(self, trainset, labels, useStdDev = False):
        logging.info("Splitting trainset into dog and cat sets")
        cats = trainset[labels == self._catlabel, :]
        dogs = trainset[labels == self._doglabel, :]
        
        logging.info("Computing means of features for dog and cat classes")
        mean_d = np.mean(dogs, axis=0)
        mean_c = np.mean(cats, axis=0)
        logging.debug("Mean of dogs: %s", str(mean_d))
        logging.debug("Mean of cats: %s", str(mean_c))
        
        logging.info("Computing absolute differences between cat and dog means")
        diffs = np.abs(mean_d - mean_c)
        logging.debug("Diff of means: %s", str(diffs))
        
        logging.info("Computing mean of absolute diffs")
        meanDiff = np.mean(diffs)
        logging.info("Mean of diffs is: %s", str(meanDiff))

        logging.info("Computing std deviation of absolute diffs")
        stdDevDiff = np.std(diffs)
        logging.info("Standard deviation of diffs is: %s", str(stdDevDiff))
        diffs = np.asarray(diffs)[0]
        
        threshold = meanDiff + stdDevDiff if useStdDev else meanDiff
        self._filter = diffs  > threshold
        
    def applyFilter(self, useStdDev = False):
        
        logging.info("Filtering data")
        self.TrainData = self._fp.TrainData[:, self._filter]
        self.ValidationData = self._fp.ValidationData[:, self._filter]
        self.TestData = self._fp.TestData[:, self._filter]
        
        self.TrainLabels = self._fp.TrainLabels
        self.ValidationLabels = self._fp.ValidationLabels
        self.TestLabels = self._fp.TestLabels
        
        logging.info("Finished filtering")
        logging.info("Features left after filtering data: %d", self.TrainData.shape[1])
