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

# Print out one of the fare
print(ride_id_li[0][5])

# Collecting fares for each driver using hashmap
driver_faredict = {}
for i in ride_id_li:
    if i[0] not in driver_faredict:
        driver_faredict[i[0]] = [i[5]]
    else:
        driver_faredict[i[0]].append(i[5])

# Print out all fares given one driver identity
print(driver_faredict['052bba06c5fc0bdea4bc2f9cb92b37c7'])

# map driver id to the ride id
driver_rideiddict = {}
for i in ride_id_li:
    if i[0] not in driver_rideiddict:
        driver_rideiddict[i[0]] = [i[1]]
    else:
        driver_rideiddict[i[0]].append(i[1])

# print ride id given a driver id
print(driver_rideiddict['007f0389f9c7b03ef97098422f902e62'])

# Reverse the dictionary
rideid_driverdict = {}
for i in driver_rideiddict.keys():
    for j in driver_rideiddict[i]:
        rideid_driverdict[j] = [i]

# print
print(rideid_driverdict['01f133164433ea7682545a41643e6949'])

# Map ride id to driver id and ride's specific time
rideid_driver_timedict = {}
for i in ride_time_li:
    if i[0] in rideid_driverdict:
        rideid_driverdict[i[0]].append(i[2])

for i in rideid_driverdict.keys():
    if len(rideid_driverdict[i]) == 2:
        rideid_driver_timedict[i] = rideid_driverdict[i]

print(rideid_driver_timedict['07f9b5246c8431e3e5bac56d9f48b4f9'])
signal = 0
for i in rideid_driver_timedict.values():
    if len(i) != 2:
        siganl += 1

# Given Ride id print out driver id and time
print(signal)

# Map driver id to all rides' time
driver_alltripsdict = {}
for i in rideid_driver_timedict.keys():
    if rideid_driver_timedict[i][0] not in driver_alltripsdict:
        driver_alltripsdict[rideid_driver_timedict[i][0]] = [rideid_driver_timedict[i][1]]
    elif rideid_driver_timedict[i][0] in driver_alltripsdict:
        driver_alltripsdict[rideid_driver_timedict[i][0]].append(rideid_driver_timedict[i][1])

# Print out all rides' time give a driver id
print(driver_alltripsdict['007f0389f9c7b03ef97098422f902e62'])

# Split the date to only fetch month and day
for i in driver_alltripsdict.keys():
    for j in range(len(driver_alltripsdict[i])):
        driver_alltripsdict[i][j] = driver_alltripsdict[i][j].split()[0].split("-")[1:]

# Print out all months and days of all rides given a driver id
print(driver_alltripsdict['007f0389f9c7b03ef97098422f902e62'])

# Change dates from string to integer
for i in driver_alltripsdict.keys():
    for j in range(len(driver_alltripsdict[i])):
        driver_alltripsdict[i][j] = [int(driver_alltripsdict[i][j][0]),int(driver_alltripsdict[i][j][1])]

# Print
print(driver_alltripsdict['052bba06c5fc0bdea4bc2f9cb92b37c7'])

'''last bording date'''
for i in driver_alltripsdict.keys():
    month = 1
    day = 1
    for j in driver_alltripsdict[i]:
        if j[0] > month:
            month = j[0]
            day = j[1]
        elif j[0] == month:
            if j[1] > day:
                day = j[1]
    driver_alltripsdict[i] = [[month, day]]

# print
print(driver_alltripsdict['052bba06c5fc0bdea4bc2f9cb92b37c7'])

# Add last boarding date
for i in driver_alltripsdict.keys():
    for k in driver_id_li:
        if i == k[0]:
            driver_alltripsdict[i].append(k[1].split()[0].split("-")[1:])
# print onboarding date and last boarding date
print(driver_alltripsdict['052bba06c5fc0bdea4bc2f9cb92b37c7'])
