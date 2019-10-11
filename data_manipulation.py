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

#Get rid of these rows in the dataset
ride_time = ride_time[ride_time.event != 'requested_at']
ride_time = ride_time[ride_time.event != 'accepted_at']
ride_time = ride_time[ride_time.event != 'picked_up_at']
ride_time = ride_time[ride_time.event != 'arrived_at']

#To numpy form
driver_id_np = (driver_id).to_numpy()
ride_id_np = (ride_id).to_numpy()
ride_time_np = (ride_time).to_numpy()

# To list
driver_id_li = (driver_id_np).tolist()
ride_id_li = (ride_id_np).tolist()
ride_time_li = (ride_time_np).tolist()

# Calculate Fare
for i in range(len(ride_id_li)):
    fare = (ride_id_li[i][2]*1.15*0.000621371 + ride_id_li[i][3]*0.22/60 + 2) * ((100+ride_id_li[i][4])/100) + 1.75
    if fare < 5:
        fare = 5
    if fare >400:
        fare = 400
    ride_id_li[i].append(fare)

print(ride_id_li[0][5])
