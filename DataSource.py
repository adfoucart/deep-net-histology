from abc import ABCMeta, abstractmethod

class DataSource:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.initialized = False

    @abstractmethod
    def setup(**kwargs):
        pass

    @abstractmethod
    def next_batch(n, **kwargs):
        pass

    @abstractmethod
    def get_inputshape():
        pass