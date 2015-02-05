import sys
import os
import logging
import copy
import itertools
from LoggingSetup import LoggingSetup
from FeatureProvider import FeatureProvider
from DataProvider import DataProvider

from FeatureExtraction.FeatureClass import FeatureExtractor
from FeatureExtraction.FeatureFilter import FeatureFilter
# from svm.TheanoSVMClassifier import SVMClassifier
# from svm.SVMClassifier import SVMClassifier
from svm.HybridSVMClassifier import SVMClassifier
from svm.SKLSVMClassifier import SKLSVMClassifier
from rf.RFClassifier import RFClassifier
from Utility import TimeManager


DEBUG = False

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
        
        self.RunSklSVM = False # run the scikit learn SVM?
        self.SklSVMArgs = {}

        self.LoadTraining = False # should we train the classifier or load the training?
        self.SaveTraining = False # save the training data?
        self.TestValidationSet = True # run a test against the validation set?
        
        self.RunFeatureFilter = False
        self.FeatureFilterArgs = {}
        self.Name = ""



class IMPLDriver(object):

    def __init__(self):
        self.FeatureProvider = None
        self.FeatureExtractor = None
        self.DataProvider = None
        self.FilteredData = None
        LoggingSetup().setup(DEBUG)
    
    def run(self, setup):
        self.Setup = setup
        self._initDataProvider()
        self._initFeatureProvider()
        
        if self.Setup.RunFeatureFilter:
            ff = FeatureFilter(self.FeatureProvider, self.Setup.DogLabel, self.Setup.CatLabel)
            ff.initFilter(**self.Setup.FeatureFilterArgs)
            ff.applyFilter()
            self.FilteredData = ff
        else:
            self.FilteredData = self.FeatureProvider
        
        if self.Setup.RunCNN:
            logging.warn("No support to run CNN through driver, yet")
            # TODO: support for CNN
            # cnn = CNN(self.DataProvider)
            # self._runClassifier(cnn, self.Setup.CNNArgs)

        if self.Setup.RunSklSVM:
            logging.info("Running SciKit Learn Support Vector Machine classifier")
            svm = SKLSVMClassifier(self.FilteredData)
            self._runClassifier(svm, self.Setup.SklSVMArgs)

        if self.Setup.RunSVM:
            logging.info("Running Support Vector Machine classifier")
            svm = SVMClassifier(self.FilteredData)
            self._runClassifier(svm, self.Setup.SVMArgs)

        if self.Setup.RunRF:
            logging.info("Running Random Forest classifier")
            rf = RFClassifier(self.FilteredData)
            self._runClassifier(rf, self.Setup.RFArgs)


    def _initFeatureProvider(self):
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
            tm = TimeManager()
            classifier.train(**classifierKwargs)
            tm.tick()
            logging.info("Training took %d:%0.2f minutes", int(tm.actual_tick/60), tm.actual_tick%60)

        if self.Setup.SaveTraining:
            logging.info("Saving training of classifier %s to file", classifier.Name)
            classifier.saveTraining()
            
        if self.Setup.TestValidationSet:
            logging.info("Testing the validation set with classifier %s", classifier.Name)
            self._runClassifierOnSet(classifier, "Train", self.FilteredData.TrainData, self.FilteredData.TrainLabels)
            self._runClassifierOnSet(classifier, "Validation", self.FilteredData.ValidationData, self.FilteredData.ValidationLabels)
            self._runClassifierOnSet(classifier, "Test", self.FilteredData.TestData, self.FilteredData.TestLabels)

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
        confname = conf.Name or "config %d" % i
        logging.info("-------------")
        logging.info("Runing: %s", confname)
        logging.info("-------------")
        print confname, ": "
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
#    loadSVMandValidate.RunFeatureFilter = True
#    loadSVMandValidate.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.8000.2000.gz")
#    loadSVMandValidate.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.8000.2000.gz")

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
    generateAndSaveFeatures.DataProviderMax = 8000
    generateAndSaveFeatures.SaveDataSetPartitioning = True
    generateAndSaveFeatures.ExtractFeatures = True
    generateAndSaveFeatures.SaveExtractedFeatures = True
    generateAndSaveFeatures.FeatureExtractionArgs = {
        'num_features' : 2000,
        'max_texel_pics' : 5000
    }

    runSVMWith8000_500 = IMPLRunConfiguration()
    runSVMWith8000_500.RunSVM = True
    runSVMWith8000_500.SaveTraining = True
    runSVMWith8000_500.CreateDataSetPartitioning = False
    runSVMWith8000_500.ExtractFeatures = False
    runSVMWith8000_500.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.8000.500.gz")
    runSVMWith8000_500.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.8000.500.gz")
    
    runSVMWithAll_1000 = IMPLRunConfiguration()
    runSVMWithAll_1000.RunSVM = True
    runSVMWithAll_1000.SaveTraining = True
    runSVMWithAll_1000.CreateDataSetPartitioning = False
    runSVMWithAll_1000.ExtractFeatures = False
    runSVMWithAll_1000.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.all.1000.gz")
    runSVMWithAll_1000.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.all.1000.gz")
    
    # this configuration doesn't extract features, but only runs the RF with training and validation test
    trainRFandValidate = IMPLRunConfiguration()
    trainRFandValidate.RunRF = True
    trainRFandValidate.SaveTraining = True
    
    # simply load the last training and run the validation test
    loadRFandValidate = IMPLRunConfiguration()
    loadRFandValidate.RunRF = True
    loadRFandValidate.LoadTraining = True
    loadRFandValidate.SaveTraining = False
    
    runRFWith8000_500 = IMPLRunConfiguration()
    runRFWith8000_500.RunRF = True
    runRFWith8000_500.SaveTraining = True
    runRFWith8000_500.CreateDataSetPartitioning = False
    runRFWith8000_500.ExtractFeatures = False
    runRFWith8000_500.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.8000.500.gz")
    runRFWith8000_500.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.8000.500.gz")

    runRFWithAll_1000 = IMPLRunConfiguration()
    runRFWithAll_1000.RunRF = True
    runRFWithAll_1000.SaveTraining = True
    runRFWithAll_1000.CreateDataSetPartitioning = False
    runRFWithAll_1000.ExtractFeatures = False
    runRFWithAll_1000.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.all.1000.gz")
    runRFWithAll_1000.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.all.1000.gz")
    
    sklVsOwnSVM = copy.copy(runSVMWithAll_1000)
    sklVsOwnSVM.DataProviderMax = 3000
    sklVsOwnSVM.SVMArgs = dict(C=10, maxIter=10, kTup=('rbf', 1.5))
    sklVsOwnSVM.RunSVM = True
    sklVsOwnSVM.RunSklSVM = False
    sklVsOwnSVM.SklSVMArgs = dict(C=10, gamma=0.0001)
    
    loadSVMandValidate.DataProviderMax = 6000
    loadSVMandValidate.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.all.1000.gz")
    loadSVMandValidate.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.all.1000.gz")   
           
    

    filterFeaturesWith8000_2000 = IMPLRunConfiguration()
    filterFeaturesWith8000_2000.SaveTraining = False
    filterFeaturesWith8000_2000.CreateDataSetPartitioning = False
    filterFeaturesWith8000_2000.ExtractFeatures = False
    filterFeaturesWith8000_2000.RunFeatureFilter = True
    filterFeaturesWith8000_2000.FeatureFilterArgs = dict(filterName="mean")
    filterFeaturesWith8000_2000.DataSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/data_segmentation.8000.2000.gz")
    filterFeaturesWith8000_2000.FeatureSavePath = os.path.join(IMPLRunConfiguration.PROJECT_BASEDIR, "saved/extracted_features.8000.2000.gz")
    
                               
    # params = itertools.product([10, 30, 60, 100], [1.1, 1.3, 1.5])
    params = itertools.product([100, 150], [1.1, 1.3, 1.5])
    # params = [[10, 1.1], [30, 1.1], [110, 2.0], [150, 1.1], [180, 1.1]]
    trainSVMConfs = []    
    for C, sigma in params:
        conf = copy.copy(filterFeaturesWith8000_2000)
        conf.DataProviderMax = 6000
        conf.RunSVM = True
        conf.RunFeatureFilter = True
        conf.SaveTraining = False
        conf.SVMArgs = dict(C=C, maxIter=10, kTup=('rbf', sigma))
        conf.Name = "Run with C=%d and sigma=%0.2f" % (C, sigma)
        trainSVMConfs.append(conf)
    
    
    rfArgs=[]
    #rfArgs.append(dict(num_attr=32, max_tries=16, subset=int(5000*2/3*0.7), min_gain=-1, thres_steps=30, forest_size=10, max_depth=12))
    #rfArgs.append(dict(num_attr=23, max_tries=23, subset=int(8000*2/3), min_gain=-1, thres_steps=30, forest_size=3, max_depth=11))
    #rfArgs.append(dict(num_attr=23, max_tries=11, subset=int(2000*0.7*2/3), min_gain=-1, thres_steps=30, forest_size=500, max_depth=7))
    rfArgs.append(dict(num_attr=32, max_tries=16, subset=int(1000*0.7*2/3), min_gain=-1, thres_steps=30, forest_size=100, max_depth=10, select_kbest=True))
    
    '''attrs = [32]#[32, 64, 125, 250, 500, 1000]
    tries = [16, 32]#[8, 16, 32]
    subsets = [466, 700] #[466, 530, 700]
    gains = [-1]#[-1, 0.01, 0.005, 0.001, 0.0005]
    steps = [15,30,45]#[15, 30]
    sizes = [200, 500]#[100, 300, 500]
    depths = [8,9,10,11,12]#[-1, 8, 10, 12, 20]
    for a in attrs:
        for t in tries:
            for sub in subsets:
                for g in gains:
                    for st in steps:
                        for si in sizes:
                            for d in depths:
                                rfArgs.append(dict(num_attr=a, max_tries=t, subset=sub, min_gain=g, thres_steps=st, forest_size=si, max_depth=d))'''
    
    
    trainRFConfs = []    
    for args in rfArgs:
        conf = copy.copy(runRFWithAll_1000)
        conf.DataProviderMax = 1000
        conf.RFArgs = args
        trainRFConfs.append(conf)

    runDriver(*trainSVMConfs)



    
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        DEBUG = True
    main()
