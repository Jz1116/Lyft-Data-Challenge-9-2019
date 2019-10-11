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

'''convert to int'''
for i in driver_alltripsdict.keys():
    for j in range(len(driver_alltripsdict[i])):
        driver_alltripsdict[i][j] = [int(driver_alltripsdict[i][j][0]),int(driver_alltripsdict[i][j][1])]
print(driver_alltripsdict['052bba06c5fc0bdea4bc2f9cb92b37c7'])

driver_on_offdict = {}
for i in driver_alltripsdict.keys():
    if len(driver_alltripsdict[i]) == 2:
        driver_on_offdict[i] = driver_alltripsdict[i]
signal = 0
for i in driver_on_offdict.keys():
    if len(driver_alltripsdict[i]) != 2:
        signal += 1

# print
print(signal)

'''calculate date difference'''
for i in driver_on_offdict.keys():
    lastMonth = driver_on_offdict[i][0][0]
    lastDay = driver_on_offdict[i][0][1]
    startMonth = driver_on_offdict[i][1][0]
    startDay = driver_on_offdict[i][1][1]

    startDays = (startMonth - 1) * 30 + startDay
    lastDays = (lastMonth - 1) * 30 + lastDay

    difference = lastDays - startDays
    driver_on_offdict[i] = difference

# print the date difference of a driver
print(driver_on_offdict['fff482c704d36a1afe8b8978d5486283'])

'''Average Lifetime Value'''
farelist = []
farelistnew = []
for i in driver_on_offdict.keys():
    for k in driver_faredict.keys():
        if i == k:
            farelist.append(driver_faredict[k])
for i in farelist:
    sum3 = 0
    for k in i:
        sum3 += k
    farelistnew.append(sum3)

# Print the total fare of a driver
print(farelistnew[0])

sum4 = 0
for i in farelistnew:
    sum4 +=  i
newaverage = sum4/len(farelistnew)

# Print average total fare
print(round(newaverage, 3))

'''Average days spent in Lyft'''
averageDay = 0
for i in driver_on_offdict.keys():
    averageDay += driver_on_offdict[i]
averageDay /= len(driver_on_offdict)

# Print out average days
print(round(averageDay, 3))

'''feature1: Driver ID'''
Driver_ID_LIST = []
for i in driver_on_offdict.keys():
    Driver_ID_LIST.append(i)

# Convert to numpy array
driver_id_nparray = np.asarray(Driver_ID_LIST).reshape(-1,1)

# Check Shape
print(driver_id_nparray.shape)

'''feature2: Days in Lyft'''
Driver_day_list = []
for i in Driver_ID_LIST:
    if i in driver_on_offdict:
        Driver_day_list.append(driver_on_offdict[i])

# Convert to numpy array
driver_days_nparray = np.asarray(Driver_day_list).reshape(-1,1)

# Check Shape
print(driver_days_nparray.shape)

"""Mix 2 features together """
a = np.append(driver_id_nparray,driver_days_nparray,axis = 1)
# Print out the numpy array
print(a)

'''Target: Life Time Value'''
LTV_dict = {}
Driver_LTV_list = []
for i in driver_on_offdict.keys():
    for j in driver_faredict.keys():
        if (i == j):
            LTV_dict[j] = driver_faredict[j]
for i in LTV_dict.keys():
    total = 0
    for j in LTV_dict[i]:
        total += j
    LTV_dict[i] = total

for i in Driver_ID_LIST:
    if i in LTV_dict:
        Driver_LTV_list.append(LTV_dict[i])

# Conver to numpy array
driver_LTV_nparrays = np.asarray(Driver_LTV_list).reshape(-1,1)

# Check Shape
print(driver_LTV_nparrays.shape)

'''feature3: Primetime Percentage for each driver'''
Driver_total_primetime = {}
for i in range(len(Driver_ID_LIST)):
    for k in range(len(ride_id_li)):
        if Driver_ID_LIST[i] == ride_id_li[k][0]:
            Driver_total_primetime[Driver_ID_LIST[i]] = []
for i in range(len(Driver_ID_LIST)):
    for k in range(len(ride_id_li)):
        if Driver_ID_LIST[i] == ride_id_li[k][0]:
            print("yes")
            Driver_total_primetime[Driver_ID_LIST[i]].append(ride_id_li[k][4])

# Print out relative level of prime time
print(Driver_total_primetime['007f0389f9c7b03ef97098422f902e62'])

for i in Driver_total_primetime.keys():
    sum1 = 0
    for k in Driver_total_primetime[i]:
        sum1 += k
    num = len(Driver_total_primetime[i])
    Driver_total_primetime[i] = sum1/num

# calculate prime time value for each driver as a unique feature
print(Driver_total_primetime['007f0389f9c7b03ef97098422f902e62'])

Driver_primetime_list = []
for i in Driver_total_primetime:
    Driver_primetime_list.append(Driver_total_primetime[i])
# Convert to numpy array
Driver_primetime_nparrays = np.asarray(Driver_primetime_list).reshape(-1,1)

# Check shape if it is consistent with others
print(Driver_primetime_nparrays.shape)

'''feature4: total ride distance'''
driver_dist_dict = {}
for i in ride_id_li:
    if i[0] not in driver_dist_dict:
        driver_dist_dict[i[0]] = [i[2]]
    else:
        driver_dist_dict[i[0]].append(i[2])


for i in driver_dist_dict.keys():
    total_dist = 0
    num_rides = 0
    for j in driver_dist_dict[i]:
        total_dist += j
        num_rides += 1
    driver_dist_dict[i] = [total_dist, num_rides]
print(driver_dist_dict["002be0ffdc997bd5c50703158b7c2491"])

total_dist_list = []
for i in Driver_ID_LIST:
    if i in driver_dist_dict:
        total_dist_list.append(driver_dist_dict[i][0])

# Convert to numpy array
total_dist_nparray = np.asarray(total_dist_list).reshape(-1,1)

# Check Shape
print(total_dist_nparray.shape)

'''feature5: total number of rides'''
total_numrides_list = []
for i in Driver_ID_LIST:
    if i in driver_dist_dict:
        total_numrides_list.append(driver_dist_dict[i][1])

total_numrides_nparray = np.asarray(total_numrides_list).reshape(-1,1)

# Check shape
print(total_numrides_nparray.shape)
