import logging
import os
import shutil
import datetime
from os import path

class LoggingSetup():
    LOGDIR = path.join(path.dirname(path.abspath(__file__)), "../logs/")
    LOGFILE_BASENAME = "{}_main.log"
    MAX_LOGFILES = 10

    def __init__(self):
        self.logpath = ""

    def setup(self, debug = False):
        """
        Set up global logging. Can then simply be used by 'import logging; logging.info("test info")'
        """
        self.logpath = self._prepareLogFile();

        level = logging.DEBUG if debug else logging.INFO 
        logging.basicConfig(filename=self.logpath, format='[%(levelname)s|%(asctime)s] %(message)s', level=level, filemode='w')



    def _prepareLogFile(self):
        """
        Make sure we have a log dir and rename existing files. Return the path to the log file to be used
        """
        logdir = path.dirname(self.LOGDIR)
        if not path.exists(logdir):
            os.mkdir(logdir)
        logfile = path.join(logdir, self.LOGFILE_BASENAME.format(datetime.date.today().isoformat()))
        self._cascadeFileIfExists(logfile)
        return logfile

    def _cascadeFileIfExists(self, basepath, level = 0):
        """
        Make sure <basepath>.<level> does not exist. If it does, rename it to <basepath>.<level+1>, recursively.
        """
        createPath = lambda l : basepath + ("" if l == 0 else ".{}".format(l))
        filename = createPath(level)
        if not path.exists(filename):
            return # nothing to do if not already existing
        self._cascadeFileIfExists(basepath, level + 1)
        cascaded = createPath(level + 1)
        # copy, don't move. we clear the file when opening. as a result we can watch logs without closing the reading application
        shutil.copy(filename, cascaded)
