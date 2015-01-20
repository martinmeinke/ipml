from LoggingSetup import LoggingSetup
from SavedFeatureProvider import SavedFeatureProvider

class IMPLDriver(object):

    def run(self):
        LoggingSetup().setup()
        featureProvider = SavedFeatureProvider()
        featureProvider.load()

if __name__ == "__main__":
    driver = IMPLDriver()
    driver.run()
