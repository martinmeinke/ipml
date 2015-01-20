'''
Created on Jan 20, 2015

@author: stefan
'''
import unittest
import logging
import os
import tempfile
from LoggingSetup import LoggingSetup


class Test(unittest.TestCase):

    def setUp(self):
        self.tempfiles = []

    def tearDown(self):
        for f in self.tempfiles:
            if os.path.exists(f):
                os.remove(f)

    def testSetup(self):
        logSetup = LoggingSetup()
        logSetup.setup()
        logging.info("test")
        logging.shutdown()
        # make sure the logfile was created
        self.assertTrue(os.path.exists(logSetup.logpath), "Log file doesn't exist")
        self.tempfiles.append(logSetup.logpath)
        # check the contents of the logfile
        with open(logSetup.logpath) as f:
            content = f.readlines()
        self.assertEqual(len(content), 1, "There is more or less input in the log than expected")
        self.assertTrue(content[0].endswith("test\n"), "Wrong message in logfile")

    def testLogCascade(self):
        logSetup = LoggingSetup()
        tempfilename = "___test_logfile"
        cascadedFilename = tempfilename + ".1"
        self.tempfiles.append(tempfilename)
        self.tempfiles.append(cascadedFilename)

        open(tempfilename, 'a').close() # create it
        logSetup._cascadeFileIfExists(tempfilename) # should rename it to <name>.1
        self.assertFalse(os.path.exists(tempfilename), "Existing file was not cascaded and still exists")
        self.assertTrue(os.path.exists(cascadedFilename), "Existing file was not cascaded to correct suffix")
        os.remove(cascadedFilename) #cleanup

if __name__ == "__main__":
    unittest.main()