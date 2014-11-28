'''
Created on Nov 28, 2014

@author: martin
'''

import glob
import unittest

def create_test_suite():
    test_file_strings = glob.glob('src/tests/test_*.py')
    module_strings = ['tests.'+str[10:len(str)-3] for str in test_file_strings]
    suites = [unittest.defaultTestLoader.loadTestsFromName(name) \
              for name in module_strings]
    
    testSuite = unittest.TestSuite(suites)

    return testSuite
