import numpy as np
import pandas as pd
import statsmodels.api as sm

class data_loader():
    # @params self._dataset: dataset that will be loaded
    # @params self._x: covariate matrix x that has to be defined
    # @params self._y: response variable y that has to be defined
    def __init__(self):
        self._dataset = None
        self._x       = None
        self._y       = None
        

    def load(self):
        raise NotImplementedError

    def x_def(self,x=[]):
        # @params x: list of covariate names ('strings'), ex: ['GPA','PSI']
        assert self._dataset is not None, "Data has not been loaded"
        self._x = self._dataset[x]

    def y_def(self,y=""):
        # @params y: string that contains the name of the variable chosen as response variable.
        assert self._dataset is not None, "Data has not been loaded"
        self._y = self._dataset[y]

    def add_constant(self): 
        assert self._x is not None, "Covariate matrix X has not been defined"
        if self._x.shape[0]>self._x.shape[1]:
            self._x.insert(0,'constant',1)
        else:
            self._x = self._x.T
            self._x.insert(0,'constant',1)
            self._x = self._x.T

    @property
    def x(self):
        # Getters for the covariate matrix x, can be called without ()
        return self._x

    @property
    def y(self):
        # Getter for the response variable y, can be called without ()
        return self._y

    def x_trans(self):
        # Transposes the covariate matrix
        self._x = self._x.T


# subclass 1:
class loader_sm(data_loader):
    def __init__(self):
        super().__init__()

    def load(self, dataset):
        # @params dataset: List of strings containing the dataset name(s)
        try:
            self._dataset = sm.datasets.get_rdataset(*dataset).data
        except ValueError:
            self._dataset = eval(f"sm.datasets.{dataset[0]}.load_pandas().data")
        except AttributeError:
            print('Dataset was not found')



# subclass 2:

class loader_int(data_loader):
    def __init__(self):
        super().__init__()

    def load(self, url):
    # @params url: url (string) of the csv file
        self._dataset = pd.read_csv(url)