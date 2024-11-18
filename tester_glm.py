from part_1 import glm_gaussian
from part_1 import glm_bernoulli
from part_1 import glm_poisson
from part_2 import loader_sm
from part_2 import loader_int
import argparse

def main(data_source, dset, url, predictors, response, add_intercept, model, split):
    
    if data_source == 'statsmodels':
        data = loader_sm()
        data.load(dset)
    elif data_source == 'internet':
        data = loader_int()
        data.load(url)

    data.x_def(predictors)
    data.y_def(response)

    if add_intercept==True:
        data.add_constant()

    data.x_trans()

    if model== 'gaussian':
        model = glm_gaussian(data.x,data.y)
    elif model== 'bernoulli':
        model = glm_bernoulli(data.x,data.y)
    elif model== 'poisson':
        model = glm_poisson(data.x,data.y)

    model.set_split(split)
    model.fit()
    model.predict()
    model.comp_mod()
    model.performance()

parser = argparse.ArgumentParser(prog='Tester program for Generalized linear models',description='Testing the performance of a given GLM on a particular dataset. See -help to see which arguments to state in the command-line. If no arguments are stated the code runs a Gaussian model on the Duncan dataset as default')
parser.add_argument('--model', choices=['gaussian','bernoulli','poisson'], default='gaussian', help='Choice of model')
parser.add_argument('--add_intercept', type=bool, default=True, help='Whether to add constant to the covariate matrix')
parser.add_argument('--split', type=float, default=0.8, help='Splits the dataset into training set (split) and test set (1-split)')
parser.add_argument('--dset',nargs='+', default=['Duncan','carData'], help='Choice of dataset')
parser.add_argument('--data_source', choices=['internet', 'statsmodels'], default='statsmodels', help='Select which source the dataset is loaded from')
parser.add_argument('--url',type=str, default='https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv', help='url must be provided if datasource is internet')
parser.add_argument('--predictors', nargs='+', default=['education','prestige'], help='Select predictor(s)')
parser.add_argument('--response', type=str, default='income', help='Select response variable')
args = parser.parse_args()

main(args.data_source, args.dset, args.url, args.predictors, args.response, args.add_intercept, args.model, args.split)