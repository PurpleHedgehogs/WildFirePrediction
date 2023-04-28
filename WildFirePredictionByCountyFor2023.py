# Source of code - https://www.kaggle.com/code/shidharthoroy/wildfires-in-california-prediction
# Modified the code
# a) to filter the data for Alameda county
# b) include prediction for year 2023 for Alameda county.
# c) compare the accuracy against for bigger training data set - all counties in State of California

# Think of a problem statement
# Acquiring Data
# Neural Network
# Training network
# Present Results
# Upload on Github

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import numpy as np
import pandas as pd
import os
import datetime as dt
import warnings


# Load Data, this step could take up to 60 seconds
weather = pd.read_csv('dataset/WeatherEvents_Jan2016-Dec2021.csv')
print ("Loaded weather data : ", weather.head(1))

isCa = weather['State'] == 'CA'
ca = weather[isCa]
print("Number of weather events in CA:", ca.size)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

start = pd.DatetimeIndex(ca['StartTime(UTC)'])
end = pd.DatetimeIndex(ca['EndTime(UTC)'])
ca['Mean_Duration'] = (end -start).total_seconds()
ca['Year'] = start.year
ca['Month'] = start.month
grouped = ca.groupby(by=['Year' ,'Month' ,'Type' ,'Severity'] ,as_index=False).agg(
    {'Mean_Duration': "mean", 'EventId': "count"})
grouped['Count'] = grouped.EventId
caGrouped = grouped[['Year', 'Month', 'Type', 'Severity', 'Count', "Mean_Duration"]]
caGrouped.head(12)
print ("Grouped data : ", caGrouped.head(12))

# Prepare weather set for machine learning
types = [{'Type': 'Cold', 'Severity': 'Severe'}, {'Type': 'Fog', 'Severity': 'Moderate'},
         {'Type': 'Fog', 'Severity': 'Severe'}, {'Type': 'Precipitation', 'Severity': 'UNK'},
         {'Type': 'Rain', 'Severity': 'Heavy'}, {'Type': 'Rain', 'Severity': 'Light'},
         {'Type': 'Rain', 'Severity': 'Moderate'}, {'Type': 'Snow', 'Severity': 'Heavy'},
         {'Type': 'Snow', 'Severity': 'Light'}, {'Type': 'Snow', 'Severity': 'Moderate'},
         {'Type': 'Storm', 'Severity': 'Severe'}]
columns = ['Count', 'Mean_Duration']


# Create weather ML input set for year which contains duration and count of different weather events for each month
def createYearSet(year, uniqueTypes):
    numberOfMonths = 12
    array = [0.0] * (numberOfMonths * len(uniqueTypes) * len(columns))

    isYear = caGrouped['Year'] == year
    caYear = caGrouped[isYear]
    i = 0
    for m in range(numberOfMonths):
        isMonth = caYear['Month'] == m + 1
        caMonth = caYear[isMonth]
        for u in uniqueTypes:
            isType = caMonth['Type'] == u['Type']
            caType = caMonth[isType]
            isSeverity = caType['Severity'] == u['Severity']
            caSevType = caType[isSeverity]
            countSize = caSevType["Count"].size
            durationSize = caSevType["Mean_Duration"].size

            if countSize == 1:
                array[i] = caSevType["Count"].item()
            i = i + 1
            if durationSize == 1:
                array[i] = caSevType["Mean_Duration"].item()
            i = i + 1

    return array


# Read wildfire events.
fire =pd.read_csv('dataset/California_Fire_Incidents.csv')
# Filter wildfire events for 2016-2019 years.
isFireRecent = fire['ArchiveYear'].isin([2016,2017,2018,2019])
fireNew = fire[isFireRecent]
fireNew.head(1)

print ("Recent Fire data : ", fireNew.head(1))

# Group fire data same way as weather data, group on month.
start = pd.DatetimeIndex(fireNew['Started']);
fireNew['Month'] = start.month
groupedFire = fireNew.groupby(by=['Counties', 'ArchiveYear','Month'] ,as_index=False).agg({'AcresBurned': "sum",'Active':'count'})
groupedFire['Count'] = groupedFire.Active
caGroupedFire = groupedFire[['ArchiveYear','Month','AcresBurned','Count','Counties']]
caGroupedFire.head(6)
print("Fire data grouped by Acres burnt : ", caGroupedFire.head(6))


# Group per year for expected outputs.
groupedFireYear = caGroupedFire.groupby(by=['Counties', 'ArchiveYear'] ,as_index=False).agg({'AcresBurned': "sum",'Count':'sum'})
mean = groupedFireYear[['AcresBurned']].mean()
groupedFireYear['BadYear'] =groupedFireYear['AcresBurned'] >  mean.item()
groupedFireYear.head(5)
print("Fire data with bad year : ", groupedFireYear.head(5))

#Filter for Alameda County

isAlamedaFire = groupedFireYear['Counties'].isin(['Alameda'])
print ("isSonomaFire : ", isAlamedaFire)
alamedaFireYear = groupedFireYear[isAlamedaFire]
print ("Sonoma data : ", alamedaFireYear[isAlamedaFire])

# Prepare expected outputs Y for ML
trainingFireYears = alamedaFireYear[alamedaFireYear['ArchiveYear'].isin([2016,2017,2018])]
trainingYTransposed = trainingFireYears.BadYear.to_frame().T.astype(float)
trainingY = np.reshape(trainingYTransposed.to_numpy(), (3,1))
print(trainingY)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Combine everything up to now and create ML model, train it, predict 2019.
input16 = createYearSet(2016,types)
input17 = createYearSet(2017,types)
input18 = createYearSet(2018,types)
input19 = createYearSet(2019,types)

input23 = createYearSet(2023,types)

firstLayerSize = len(input16)
secondLayerSize = 90
thirdLayerSize = 40
outputSize = 1

print("Setting up model")
print("Creating Input layer with size : "+ str(len(input16)))
inputs = keras.Input(shape=(len(input16),))
dense1 = layers.Dense(firstLayerSize, activation="sigmoid")(inputs)
dense2 = layers.Dense(secondLayerSize, activation="relu")(dense1)
dense3 = layers.Dense(thirdLayerSize, activation="relu")(dense2)
outputs = layers.Dense(outputSize, activation="sigmoid")(dense3)

model = keras.Model(inputs=inputs, outputs=outputs, name="learnModel")
model.summary()

opt = keras.optimizers.SGD(lr=0.05, momentum=0.5)
model.compile(loss='mean_squared_error', optimizer=opt,metrics=["accuracy"])

trainingX = np.array([input16,input17,input18])
history = model.fit(trainingX,trainingY,batch_size=3, epochs=40)

test_scores = model.evaluate(trainingX, trainingY, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# We predict for 2019, we know this is not a 'bad' year and should be predicted with a low value (smaller than 0.5)
# This model is specialized in only three years and probably has quite some variance (overfitting problem) when applied to other years.
model.predict([input19])

print ("In Alameda County, 2023 will be a", "bad year with a lot of wildfires" if (model.predict([input23]) > 0.5) else "good year with little wildfires")
