import logging
from os import path
from LoggingSetup import LoggingSetup
from SavedFeatureProvider import SavedFeatureProvider
from DataProvider import DataProvider

from svm.SVMClassifier import SVMClassifier

class IMPLRunConfiguration(object):
    
    def __init__(self):
        self.CreateDataSetPartitioning = False # create a new train/validation data segmentation or load the existing
        self.DataSegmentation = (0.6, 0.2, 0.2) # how much of the labeled data should be used as the train/validation/test set 
        self.SaveDataSetPartitioning = False # should we save the (probably newly created) data segmentation or not
        self.RawDataDir = DataProvider.RAWDATADIR # where we can find the raw data
        self.CatDataPrefix = DataProvider.CAT_DATAPREFIX # the prefix of the cat files
        self.DogDataPrefix = DataProvider.DOG_DATAPREFIX # the prefix of the dog files
        
        self.ExtractFeatures = False # should we extract the features or load from file
        self.SaveExtractedFeatures = False # if the features were extracted: should we save them to file
        self.SavedCatFeaturesFile = SavedFeatureProvider.CAT_FEATURES_FILE # file with cat features to be loaded
        self.SavedDogFeaturesFile = SavedFeatureProvider.DOG_FEATURES_FILE # file with dog features to be loaded
        self.SavedExtractorDataFile = SavedFeatureProvider.EXTRACTOR_DATA_FILE  # file with extractor data to be loaded
        self.FeatureExtractionArgs = {} # args for the feature extractor

        # Actually, the important parameters
        self.RunCNN = False # run the convolutional neural network?
        self.CNNArgs = {} # arguments for the convolutional neural network

        self.RunSVM = False # run the support vector machine?
        self.SVMArgs = {} # arguments for the support vector machine

        self.RunRF = False # run the random forest?
        self.RFArgs = {} # arguments for the random forest

        self.LoadTraining = False # should we train the classifier or load the training?
        self.SaveTraining = False # save the training data?
        self.TestValidationSet = True # run a test against the validation set?


class IMPLDriver(object):

    def __init__(self):
        self.FeatureProvider = None
        self.FeatureExtractor = None
        self.DataProvider = None
    
    def run(self, setup):
        self.Setup = setup
        LoggingSetup().setup()
        self._initDataProvider()
        self._initFeatureProvider()
        
        if self.Setup.RunCNN:
            logging.warn("No support to run CNN through driver, yet")
            # TODO: support for CNN
            # cnn = CNN(self.DataProvider)
            # self._runClassifier(cnn, self.Setup.CNNArgs)

        if self.Setup.RunSVM:
            logging.info("Running Support Vector Machine classifier")
            svm = SVMClassifier(self.FeatureProvider)
            self._runClassifier(svm, self.Setup.SVMArgs)

        if self.Setup.RunRF:
            logging.warn("No support to run RF classifier through driver, yet")
            # TODO: support for RF 
            # logging.info("Running Random Forest classifier")
            # rf = RFClassifier(self.FeatureProvider)
            # self._runClassifier(rf, self.Setup.RFArgs)


    def _initFeatureProvider(self):
        # check if init is necessary
        if not self.Setup.RunSVM and not self.Setup.RunRF and not self.Setup.ExtractFeatures:
            logging.info("Not initializing a feature provider - SVM and RF won't run")
            return
        if self.FeatureProvider:
            logging.info("FeatureProvider already initialized")
            return

        # check if we should load from file
        featurefiles = (self.Setup.SavedCatFeaturesFile, self.Setup.SavedDogFeaturesFile, self.Setup.SavedExtractorDataFile)
        if not self.Setup.ExtractFeatures:
            logging.info("Loading saved features from '%s', '%s', and '%s'", *featurefiles)
            self.FeatureProvider = SavedFeatureProvider(*featurefiles)
            logging.info("Start to load SavedFeatureProvider")
            self.FeatureProvider.load(self.Setup.DataSegmentation[1]) # TODO: remove this workaround
            return

        # shit's getting real, we extract the features as we go!
        logging.warn("No support for extracting features from the driver, yet.") # for now
        self.FeatureProvider = None
        # TODO: support for the "real" feature extractor 
        # self.FeatureExtractor = FeatureExtractor()
        # self.FeatureProvider = FeatureProvider(_self.DataProvider, self.FeatureExtractor)
        # logging.info("Load the FeatureExtractor based FeatureProvider")
        # self.FeatureProvider.load(self.Setup.FeatureExtractionArgs)

        if self.Setup.SaveExtractedFeatures:
            logging.info("Saving extracted features to files '%s', '%s', and '%s'", *featurefiles)
            logging.warn("No support for saving features, yet.") # for now
            # self.FeatureProvider.save(*featurefiles)


    def _initDataProvider(self):
        if not self.Setup.RunSVM and not self.Setup.RunRF and not self.Setup.ExtractFeatures and not self.Setup.CreateDataSetPartitioning:
            logging.info("Not initializing a feature provider - SVM, RF, and FeatureExtraction won't run")
            return
        if self.DataProvider:
            logging.info("DataProvider already initialized")
            return
        self.DataProvider = DataProvider(self.Setup.RawDataDir, self.Setup.CatDataPrefix, self.Setup.DogDataPrefix)

        # create new data segmentation or load from file?
        if self.Setup.CreateDataSetPartitioning:
            logging.info("Loading data provider with validation data portion of {}".format(self.Setup.DataSegmentation))
            self.DataProvider.load(self.Setup.DataSegmentation)
        else:
            logging.info("Loading data provider from file")
            self.DataProvider.loadFromFile()

        # save to file if we want to
        if self.Setup.SaveDataSetPartitioning:
            logging.info("Save DataProvider to file")
            self.DataProvider.saveToFile()

    def _runClassifier(self, classifier, classifierKwargs):
        if self.Setup.LoadTraining:
            logging.info("Loading training for classifier %s from file", classifier.Name)
            classifier.loadTraining()
        else:
            logging.info("Training the classifier %s", classifier.Name)
            classifier.train(**classifierKwargs)

        if self.Setup.SaveTraining:
            logging.info("Saving training of classifier %s to file", classifier.Name)
            classifier.saveTraining()
            
        errorRate = -1
        if self.Setup.TestValidationSet:
            logging.info("Testing the validation set with classifier %s", classifier.Name)
            # TODO: which output do we return. and how?
            errorRate = classifier.testValidationSet()
            msg = "%s: Validation Test Error Rate = %s" % (classifier.Name, str(errorRate))
            logging.info("RESULT of %s", msg)
            print msg
        return errorRate
        # TODO: well we should now actually call the classifier to check a test set
        # classifier.classify()



if __name__ == "__main__":
    # this configuration doesn't extract features, but only runs the SVM with training and validation test
    trainSVMandValidate = IMPLRunConfiguration()
    trainSVMandValidate.RunSVM = True
    trainSVMandValidate.SaveTraining = True
    
    # simply load the last training and run the validation test
    loadSVMandValidate = IMPLRunConfiguration()
    loadSVMandValidate.RunSVM = True
    loadSVMandValidate.LoadTraining = True
    loadSVMandValidate.SaveTraining = False
    

    # don't run a classifier, just extract the features and save them to file
    extractAndSaveFeatures = IMPLRunConfiguration()
    extractAndSaveFeatures.ExtractFeatures = True
    extractAndSaveFeatures.SaveExtractedFeatures = True
    extractAndSaveFeatures.FeatureExtractionArgs = {}
    
    # create and save a data segmentation
    segmentDataAndSaveSegmentation = IMPLRunConfiguration()
    segmentDataAndSaveSegmentation.ExtractFeatures = False
    segmentDataAndSaveSegmentation.DataSegmentation = (70, 15, 15)
    segmentDataAndSaveSegmentation.CreateDataSetPartitioning = True
    segmentDataAndSaveSegmentation.SaveDataSetPartitioning = True

    driver = IMPLDriver()
    driver.run(trainSVMandValidate)
