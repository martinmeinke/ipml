import os
import logging
import copy
from LoggingSetup import LoggingSetup
from FeatureProvider import FeatureProvider
from DataProvider import DataProvider

from FeatureExtraction.FeatureClass import FeatureExtractor
from svm.SVMClassifier import SVMClassifier
from rf.RFClassifier import RFClassifier

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
        LoggingSetup().setup()
    
    def run(self, setup):
        self.Setup = setup
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
            logging.info("Running Random Forest classifier")
            rf = RFClassifier(self.FeatureProvider)
            self._runClassifier(rf, self.Setup.RFArgs)


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
            self.FeatureProvider.loadFromFile(self.Setup.FeatureSavePath, self.Setup.DataProviderMax)

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
            self.DataProvider.loadFromFile(self.Setup.DataSavePath, self.Setup.DataProviderMax)

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
            
        if self.Setup.TestValidationSet:
            logging.info("Testing the validation set with classifier %s", classifier.Name)
            self._runClassifierOnSet(classifier, "Validation", self.FeatureProvider.ValidationData, self.FeatureProvider.ValidationLabels)
            self._runClassifierOnSet(classifier, "Test", self.FeatureProvider.TestData, self.FeatureProvider.TestLabels)

    def _runClassifierOnSet(self, classifier, runname, data, labels):
        errorRate = classifier.testDataSet(data, labels)
        msg = "%s: %s Error Rate = %s" % (runname, classifier.Name, str(errorRate))
        logging.info("RESULT of %s", msg)
        print msg

def runDriver(*configurations):
    driver = IMPLDriver()
    logging.info("Running %d different configuration(s)" % len(configurations))
    i = 0
    for conf in configurations:
        logging.info("-------------")
        logging.info("Run config %d", i)
        logging.info("-------------")
        # log exceptions and throw them again
        try:
            driver.run(conf)
        except Exception as e:
            logging.exception(str(e))
        i += 1

def main():
    # this configuration doesn't extract features, but only runs the SVM with training and validation test
    trainSVMandValidate = IMPLRunConfiguration()
    trainSVMandValidate.RunSVM = True
    trainSVMandValidate.SaveTraining = True
    trainSVMandValidate.DataProviderMax = 5000
    
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

    runSVMWith8000_500 = IMPLRunConfiguration()
    runSVMWith8000_500.RunSVM = True
    runSVMWith8000_500.SaveTraining = True
    runSVMWith8000_500.CreateDataSetPartitioning = False
    runSVMWith8000_500.ExtractFeatures = False
    runSVMWith8000_500.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.8000.500.gz")
    runSVMWith8000_500.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.8000.500.gz")
    
    
    # this configuration doesn't extract features, but only runs the RF with training and validation test
    trainRFandValidate = IMPLRunConfiguration()
    trainRFandValidate.RunRF = True
    trainRFandValidate.SaveTraining = True
    
    # simply load the last training and run the validation test
    loadRFandValidate = IMPLRunConfiguration()
    loadRFandValidate.RunRF = True
    loadRFandValidate.LoadTraining = True
    loadRFandValidate.SaveTraining = False
    
    svmArgs = [
        dict(C=5, maxIter=5, kTup=('rbf', 1.3)),
        dict(C=10, maxIter=5, kTup=('rbf', 1.3)),
        dict(C=15, maxIter=5, kTup=('rbf', 1.3)),
        dict(C=20, maxIter=5, kTup=('rbf', 1.3)),
    ]
    trainSVMConfs = []    
    for args in svmArgs:
        conf = copy.copy(runSVMWith8000_500)
        conf.DataProviderMax = 5000
        conf.SVMArgs = args
        trainSVMConfs.append(conf)

    runDriver(*trainSVMConfs)
    
if __name__ == "__main__":
    main()
