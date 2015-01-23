import os
import logging
from LoggingSetup import LoggingSetup
from FeatureProvider import FeatureProvider
from DataProvider import DataProvider

from FeatureExtraction.FeatureClass import FeatureExtractor
from svm.SVMClassifier import SVMClassifier

class IMPLRunConfiguration(object):
    PROJECT_BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    
    def __init__(self):
        self.CreateDataSetPartitioning = False # create a new train/validation data segmentation or load the existing
        self.SaveDataSetPartitioning = False # should we save the (probably newly created) data segmentation or not
        self.DataSegmentation = DataProvider.DEFAULT_SEGMENTATION # how much of the labeled data should be used as the train/validation/test set
        self.DataProviderMax = DataProvider.MAXFILES # upper limit of files to be taken into consideration
        self.RawDataDir = DataProvider.RAWDATADIR # where we can find the raw data
        self.CatDataPrefix = DataProvider.CAT_DATAPREFIX # the prefix of the cat files
        self.DogDataPrefix = DataProvider.DOG_DATAPREFIX # the prefix of the dog files
        self.DataSavePath = DataProvider.SAVEPATH
        self.CatLabel = DataProvider.CAT_LABEL # int label used for cat data
        self.DogLabel = DataProvider.DOG_LABEL # int label used for dog data
        
        self.ExtractFeatures = False # should we extract the features or load from file
        self.SaveExtractedFeatures = False # if the features were extracted: should we save them to file
        self.FeatureExtractionArgs = {} # args for the feature extractor
        self.FeatureSavePath = FeatureProvider.SAVEPATH

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

        self.FeatureExtractor = FeatureExtractor(**self.Setup.FeatureExtractionArgs)
        self.FeatureProvider = FeatureProvider(self.DataProvider, self.FeatureExtractor)
        logging.info("Load the FeatureExtractor based FeatureProvider")
        if self.Setup.ExtractFeatures:
            # shit's getting real, we extract the features as we go!
            self.FeatureProvider.initialize()
        else:
            # we chill and load stuff simply from file
            self.FeatureProvider.loadFromFile(self.Setup.FeatureSavePath)

        if self.Setup.SaveExtractedFeatures:
            logging.info("Saving extracted features to file")
            self.FeatureProvider.saveToFile(self.Setup.FeatureSavePath)


    def _initDataProvider(self):
        if not self.Setup.RunSVM and not self.Setup.RunRF and not self.Setup.ExtractFeatures and not self.Setup.CreateDataSetPartitioning:
            logging.info("Not initializing a feature provider - SVM, RF, and FeatureExtraction won't run")
            return
        if self.DataProvider:
            logging.info("DataProvider already initialized")
            return
        self.DataProvider = DataProvider(self.Setup.CatLabel, self.Setup.DogLabel, self.Setup.RawDataDir, self.Setup.CatDataPrefix, self.Setup.DogDataPrefix)

        # create new data segmentation or load from file?
        if self.Setup.CreateDataSetPartitioning:
            logging.info("Loading data provider with validation data portion of {}".format(self.Setup.DataSegmentation))
            self.DataProvider.initialize(self.Setup.DataSegmentation, self.Setup.DataProviderMax)
        else:
            logging.info("Loading data provider from file")
            self.DataProvider.loadFromFile(self.Setup.DataSavePath)

        # save to file if we want to
        if self.Setup.SaveDataSetPartitioning:
            logging.info("Save DataProvider to file")
            self.DataProvider.saveToFile(self.Setup.DataSavePath)

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
        # TODO: do the same as validation but with test set. so classifier.testTestSet()



def main():
    # this configuration doesn't extract features, but only runs the SVM with training and validation test
    trainSVMandValidate = IMPLRunConfiguration()
    trainSVMandValidate.RunSVM = True
    trainSVMandValidate.SaveTraining = True
    
    # simply load the last training and run the validation test
    loadSVMandValidate = IMPLRunConfiguration()
    loadSVMandValidate.RunSVM = True
    loadSVMandValidate.LoadTraining = True
    loadSVMandValidate.SaveTraining = False
    

    # create and save a data segmentation
    segmentDataAndSaveSegmentation = IMPLRunConfiguration()
    segmentDataAndSaveSegmentation.ExtractFeatures = False
    segmentDataAndSaveSegmentation.CreateDataSetPartitioning = True
    segmentDataAndSaveSegmentation.SaveDataSetPartitioning = True
    
    # just make a few features for testing an save them
    testExtractSomeFeaturesAndSave = IMPLRunConfiguration()
    testExtractSomeFeaturesAndSave.CreateDataSetPartitioning = True
    testExtractSomeFeaturesAndSave.DataProviderMax = 200 # just consider 200 files
    testExtractSomeFeaturesAndSave.SaveDataSetPartitioning = True
    testExtractSomeFeaturesAndSave.ExtractFeatures = True
    testExtractSomeFeaturesAndSave.SaveExtractedFeatures = True
    testExtractSomeFeaturesAndSave.FeatureExtractionArgs = {
        'num_features' : 100 # just consider 100 features
    }
    
    # test segmentation, extraction with few features and run svm
    testSegmentExtractSVM = IMPLRunConfiguration()
    testSegmentExtractSVM.CreateDataSetPartitioning = True
    testSegmentExtractSVM.DataProviderMax = 200 # just consider 200 files
    testSegmentExtractSVM.SaveDataSetPartitioning = True
    testSegmentExtractSVM.ExtractFeatures = True
    testSegmentExtractSVM.SaveExtractedFeatures = True
    testSegmentExtractSVM.FeatureExtractionArgs = {
        'num_features' : 100 # just consider 100 features
    }
    testSegmentExtractSVM.RunSVM = True
    testSegmentExtractSVM.SaveTraining = True
    
    # generate some real and useful features
    generateAndSaveFeatures = IMPLRunConfiguration()
    generateAndSaveFeatures.CreateDataSetPartitioning = True
    generateAndSaveFeatures.DataProviderMax = 8000 # num files to take into consideration
    generateAndSaveFeatures.SaveDataSetPartitioning = True
    generateAndSaveFeatures.ExtractFeatures = True
    generateAndSaveFeatures.SaveExtractedFeatures = True
    generateAndSaveFeatures.FeatureExtractionArgs = {
        'num_features' : 500,
        'max_texel_pics' : 5000
    }


    driver = IMPLDriver()
    # log exceptions and throw them again
    try:
        driver.run(trainSVMandValidate)
    except Exception as e:
        logging.exception(str(e))
        raise

if __name__ == "__main__":
    main()
