import logging
import logging.handlers
import datetime

class MyLogger(object):
    def __init__(self, loggername, fmt=None):
        super(MyLogger, self).__init__()
        if fmt is not None:
            self.fmt = fmt
        else:
            self.fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(logging.DEBUG)
        debug_handler = logging.FileHandler('./res/%s.info'%loggername)
        debug_handler.setFormatter(logging.Formatter("%(asctime)-15s %(process)d %(message)s"))
        info_handler = logging.FileHandler('./res/%s.error'%loggername)
        info_handler.setLevel(logging.ERROR)
        info_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)

    def info(self, msg):
        self.logger.info(msg)
        return

    def debug(self, msg):
        self.logger.debug(msg)
        return

    def warnning(self, msg):
        self.logger.warning(msg)
        return

    def error(self, msg):
        self.logger.error(msg)
        return

    def critical(self, msg):
        self.logger.critical(msg)
        return
