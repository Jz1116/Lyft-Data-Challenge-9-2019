import pandas as pd
from pandas.plotting import radviz
import matploblib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#import the model we are using

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

def read(file):
    return pd.read_csv(file)

driver_id = read('driver_ids.csv')
ride_id = read('ride_ids.csv')
ride_time = read('ride_timestamps.csv')
