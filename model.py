# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:03:16 2023

@author: Alex Hoang
"""

import pandas as pd
# Describe data
data = pd.read_csv("./Dataset/KSI.csv")

data.describe()

data.head()

data.info()

data.shape

# Statistical assessment
data.mean()

data.median()

data.corr()

# Check missing values
data.isnull().sum()

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FuncFormatter)

# To solve the problem of overlapping labels on x-axis
def wrap_labels(labels, width):
    wrapped_labels = []
    for label in labels:
        wrapped_label = '\n'.join([label[i:i+width] for i in range(0, len(label), width)])
        wrapped_labels.append(wrapped_label)
    return wrapped_labels

data.hist(figsize=(15, 20))

data['TIME'].hist()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Distribution of Time')
plt.show()

class_counts = data['INJURY'].value_counts() # data classes are really imbalanced
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Injury')
plt.ylabel('Count')
plt.title('Distribution of Classes')
plt.show()

cross_tab = pd.crosstab(data['VEHTYPE'], data['INJURY'])
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cross_tab, cmap='Blues', linewidths=0.5, linecolor="black", annot=True)
plt.xlabel('INJURY')
plt.ylabel('VEHICLE TYPE')
plt.title('Heatmap of INJURY and Vehicle Type')
plt.show()

year_counts = data['YEAR'].value_counts()
plt.bar(year_counts.index, year_counts.values, width=0.5)
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Distribution of Accidents over Years')
plt.show()

street1_counts = data['STREET1'].value_counts().head(5)
plt.bar(wrap_labels(street1_counts.index, 9), street1_counts.values, width=0.5, align='center')
plt.xlabel('Street')
plt.ylabel('Number of Accidents')
plt.title('Top 5 Streets with most Accidents')
plt.show()

plt.scatter(data['LONGITUDE'], data['LATITUDE'])
plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.title('Accidents based on Longtitude and Latitude')
plt.show()

visibility_counts = data['VISIBILITY'].value_counts()
fig, ax = plt.subplots(figsize=(10,10))
plt.bar(wrap_labels(visibility_counts.index, 11), visibility_counts.values, width=0.8)
plt.xlabel('Visibility')
plt.ylabel('Number of Accidents')
plt.title('Visibility Conditions over Accidents')
plt.show()

drivact_counts = data['DRIVACT'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(15,15))
plt.bar(wrap_labels(drivact_counts.index, 16), drivact_counts.values, width=0.8)
plt.xlabel('Driver Action')
plt.ylabel('Number of Accidents')
plt.title('Top 8 Driver Action in most Accidents')
plt.show()