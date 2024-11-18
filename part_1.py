# super class GLM
from scipy.optimize import minimize
from scipy.stats import norm, bernoulli, poisson
import numpy as np
import statsmodels.api as sm
import pandas as pd

class glm():
    # @params
    def __init__(self,X,Y):
        # @params X: the covariate matrix x, already transposed
        # @params Y: Response variable
        self._x = X
        self._y = Y
        self._params = []         # Our model parameters.
        self._y_test = []         # Our predicted y response variable.
        self._comp_params = []    # The model parameters obtained in the comparison sm model.
        self._comp_y_test = []    # The predicted y response variable in the comparison sm model.
        self._x_train = []        # the training set of covariates.
        self._y_train = []        # The training set of the response variable.
        self._x_test = []         # the test set of covariates.
        self._y_val = []          # The true y for the test set, to evaluate the performance of our predictions.

    def set_split(self,split):
        # @params split: split decides the portion of the dataset that goes in the training set. 1-split goes to the test set
        assert 1>split>0, 'Invalid split parameter, must be [0,1]'
        train = int(split*len(self._y))
        self._x_train = self._x.iloc[:,0:train]
        self._y_train = self._y.iloc[0:train]
        self._x_test = self._x.iloc[:,train:]
        self._y_val = self._y.iloc[train:]
        

    def params(self):
        return self._params

    def _negloglik(self):
        raise NotImplementedError

    def fit(self):
        # Fits the model on the training data using maximum likelihood and a given neg-log-likelihood function 
        # @params init_params: initial guess for the minimization algorithm.
        init_params = np.repeat(0.1 , self._x.shape[0])
        results = minimize(self._negloglik, init_params, args =(self._x_train , self._y_train))
        self._params = results['x']       
        
    def predict(self):    
        raise NotImplementedError

    def comp_model(self):
        raise NotImplementedError

    def _r_squared(self,y_val,y_test):
        # Helper method to calculate the R-squared
        RSS = np.sum((y_val-y_test)**2)
        TSS = np.sum((y_val-np.mean(y_val))**2)
        r_squared = 1-(RSS/TSS)
        return r_squared

    def performance(self):
        # Provides a table with the parameter estimates and the R-squared of our model vs the sm model
        model_r_squared = self._r_squared(self._y_val, self._y_test)
        comp_model_r_squared = self._r_squared(self._y_val, self._comp_y_test)
        variable_names = list(self._x.index)
        variable_names.append("R-squared")

        our_model_params = list(self._params) + [model_r_squared]
        comp_model_params = list(self._comp_params) + [comp_model_r_squared]

        comparison_table = pd.DataFrame({"Our Model": our_model_params,"Comparison SM Model": comp_model_params},index=variable_names)

        print("\nComparison Table with parameter estimates and OOS R-squared:")
        print(comparison_table)

        
class glm_gaussian(glm):
    # subclass to be used when y is assumed to be normally distributed
    
    def __init__(self, X,Y):
        super().__init__(X,Y)

    def _negloglik(self,params,x,y):
        # @params params: the model parameters in the optimizaton process
        # @params x: covariate matrix x
        # @params y: Response variable
        eta = np.matmul(np.transpose(x), params)
        # identity link function
        mu = eta
        negloglik = -np.sum(norm.logpdf(y, mu))
        return negloglik

    def predict(self):
        # predicts the gaussian response variable based on a test set of covariates and the parameters from the fit() method
        for i in range(self._x_test.shape[1]):
            self._y_test.append(np.matmul(self._params,self._x_test.iloc[:,i]))

    def comp_mod(self):
        # Provides the Gaussian GLM from the statsmodels library to used for comparison
        comp_mod = sm.GLM(self._y_train,self._x_train.T,family = sm.families.Gaussian()).fit()
        self._comp_params = comp_mod.params
        self._comp_y_test = comp_mod.predict(self._x_test.T)


class glm_bernoulli(glm):
    #subclass to be used when the y is assumed to be bernoulli distributed
    def __init__(self, X,Y):
        super().__init__(X,Y)

    def _negloglik(self,params,x,y):
        eta = np.matmul(np.transpose(x), params)
        # identity link function
        mu = np.exp(eta)/(1+np.exp(eta))
        negloglik = -np.sum(bernoulli.logpmf(y, mu))
        return negloglik

    def predict(self):
        # predicts the bernoulli distributed response variable based on a test set of covariates and the parameters from the fit() method
        for i in range(self._x_test.shape[1]):
            eta = np.matmul(self._params,self._x_test.iloc[:,i])
            self._y_test.append(np.exp(eta)/(1+np.exp(eta)))

    def comp_mod(self):
        # Provides the Bernoulli GLM from the statsmodels library to used for comparison
        comp_mod = sm.GLM(self._y_train,self._x_train.T,family = sm.families.Binomial()).fit()
        self._comp_params = comp_mod.params
        self._comp_y_test = comp_mod.predict(self._x_test.T)


class glm_poisson(glm):
    # subclass to be used when y is assumed to be poisson distributed
    def __init__(self, X,Y):
        super().__init__(X,Y)

    def _negloglik(self,params,x,y):
        eta = np.matmul(np.transpose(x), params)
        # identity link function
        mu = np.exp(eta)
        negloglik = -np.sum(poisson.logpmf(y, mu))
        return negloglik

    def predict(self):
        # predicts the Poisson distributed response variable based on a test set of covariates and the parameters from the fit() method
        for i in range(self._x_test.shape[1]):
            eta = np.matmul(self._params,self._x_test.iloc[:,i])
            self._y_test.append(np.exp(eta))

    def comp_mod(self):
        # Provides the Poisson GLM from the statsmodels library to used for comparison
        comp_mod = sm.GLM(self._y_train,self._x_train.T,family = sm.families.Poisson()).fit()
        self._comp_params = comp_mod.params
        self._comp_y_test = comp_mod.predict(self._x_test.T)
