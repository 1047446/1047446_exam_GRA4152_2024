from part_1 import glm_gaussian
from part_1 import glm_bernoulli
from part_1 import glm_poisson
from part_2 import loader_sm
from part_2 import loader_int
import argparse

def main(data_source, dset):
    
    if data_source == 'statsmodels':
        data = loader_sm()
        data.load(dset)

    print(dset)


parser = argparse.ArgumentParser(prog='Tester program for Generalized linear models',description='Testing the performance of a given GLM on a particular dataset')
parser.add_argument('--dset', default=['Duncan','carData'], help='Choice of dataset')
parser.add_argument('--data_source', choices=['internet', 'statsmodels'], default='statsmodels', help='Select which source the dataset is loaded from')

args = parser.parse_args()

main(args.data_source,args.dset)