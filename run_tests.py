'''
Created on Nov 28, 2014

@author: martin
'''

import unittest
import tests.all_tests

if __name__ == '__main__':
    testSuite = tests.all_tests.create_test_suite()
    text_runner = unittest.TextTestRunner().run(testSuite)