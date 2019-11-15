import time
import pickle
import argparse
# set seed to reproduce results with tf and keras
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
# other imports
from keras.layers import Input, Dense, multiply, Lambda
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras import optimizers
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint



def train_algo(algo, algo_name, train_data):
    print("Training algorithm: " + algo_name)
    
    #train algorithm
    algo.fit( train_data )
    
    # save the model to disk
    filename = "models/cb12_"  + algo_name + ".model"
    print("Finished training. Storing model to: " + filename)

    pickle.dump(algo, open(filename, 'wb'), protocol=4)
    pass

if __name__ == '__main__':
    path =  "../../data/"
    dataset = "cb12/"

    raw_path = path + dataset + "raw/" 
    interim_path = path + dataset + "interim/"
    processed_path = path + dataset + "processed/"


    valid_train = pd.read_csv(processed_path + "valid_train_14d.csv", header=0, sep='\t')
    valid_test = pd.read_csv(processed_path + "valid_test_14d.csv", header=0, sep='\t')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", 
        type=str, 
        help="Algorithm to train (ae|dae|vae)", 
        nargs='?', 
        default="ae", 
        const="ae")
    
    args = parser.parse_args()
    
