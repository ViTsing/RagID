# calculate error type distribution of each dataset
from distribution.sighan_error import calculateDistribution
from distribution.process import processInit
import os

def getDistributionsFromCSVFile(output_file): 
    '''
    param:
        output_file: File used to store the error distribution of the dataset
    function:
        Get the distribution of the data set. The distribution calculation tool comes from the author of the paper <Towards Better Chinese Spelling Check for Search Engines: A New Dataset and Strong Baseline>
    '''

    calculateDistribution(output_file)


output_file="datasetErrorDistirbution.csv"
flag_file_path = './distribution/token_replace_dict/flag.txt'
    
# Check if flag.txt file exists
if not os.path.exists(flag_file_path):
    processInit()
getDistributionsFromCSVFile(output_file)



