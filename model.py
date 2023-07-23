# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:03:16 2023

@author: Alex Hoang
@author: Aaron Yi-Lin Lou (301226659)
"""

from collections import Counter

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FuncFormatter)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics.pairwise import _euclidean_distances
# used to balance data by oversampling using KNN
from imblearn.over_sampling import SMOTE
import numpy as np

# Describe data
os.getcwd()

# define the path that locates the folder
path = r'C:/Users/lou22/Documents/SoftwareEngineeringEducation/ClassesSummer2023/COMP247SupLearn/Collision-Injury-Predictor'
os.chdir(path)
data = pd.read_csv(r"./Dataset/KSI.csv")


print("Column names:\n")
print(data.columns.values)
print("Data shape: \n")
print(data.shape)
print("Describe the data: \n")
print(data.describe(include='all'))
print("Column data types: \n")
print(data.dtypes)
print("First 5 records: \n")
for i, j in data.head(5).iterrows():
    print(i, j)
    print('------------')
print("Data information: \n")
print(data.info())

# Statistical assessment
print("Mean: \n")
print(data.mean())

print("Median: \n")
print(data.median())

print("Correlation: \n")
print(data.corr())

# Check missing values
print("Null values: \n")
print(data.isnull().sum())

# Check unique values
print("Unique values for each column: \n")
for col in data.columns:
    print(f'{col}: {data[col].nunique()}')


# Data Visualization

# To solve the problem of overlapping labels on x-axis

def wrap_labels(labels, width):
    wrapped_labels = []
    for label in labels:
        wrapped_label = '\n'.join([label[i:i+width]
                                  for i in range(0, len(label), width)])
        wrapped_labels.append(wrapped_label)
    return wrapped_labels


data.hist(figsize=(15, 20))

data['TIME'].hist()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Distribution of Time')
plt.show()

# data classes are really imbalanced
class_counts = data['INJURY'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Injury')
plt.ylabel('Count')
plt.title('Distribution of Classes')
plt.show()

cross_tab = pd.crosstab(data['VEHTYPE'], data['INJURY'])
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cross_tab, cmap='Blues', linewidths=0.5,
            linecolor="black", annot=True)
plt.xlabel('INJURY')
plt.ylabel('VEHICLE TYPE')
plt.title('Heatmap of INJURY and Vehicle Type')
plt.show()

cross_tab = pd.crosstab(data['ACCLOC'], data['ACCLASS'])
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cross_tab, cmap='Blues', linewidths=0.5,
            linecolor="black", annot=True)
plt.xlabel('FATALITY')
plt.ylabel('Accident Location')
plt.title('Heatmap of Fatality and Accident Location')
plt.show()

year_counts = data['YEAR'].value_counts()
plt.bar(year_counts.index, year_counts.values, width=0.5)
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Distribution of Accidents over Years')
plt.show()

street1_counts = data['STREET1'].value_counts().head(5)
plt.bar(wrap_labels(street1_counts.index, 9),
        street1_counts.values, width=0.5, align='center')
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
fig, ax = plt.subplots(figsize=(10, 10))
plt.bar(wrap_labels(visibility_counts.index, 11),
        visibility_counts.values, width=0.8)
plt.xlabel('Visibility')
plt.ylabel('Number of Accidents')
plt.title('Visibility Conditions over Accidents')
plt.show()

drivact_counts = data['DRIVACT'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(15, 15))
plt.bar(wrap_labels(drivact_counts.index, 16), drivact_counts.values, width=0.8)
plt.xlabel('Driver Action')
plt.ylabel('Number of Accidents')
plt.title('Top 8 Driver Action in most Accidents')
plt.show()

"""
Data modeling
"""


"""
Transformation:


Extract features: Date-> year, month, day of the week, time
time -> rush hour or not rush hour

target: fatal -> boolean
street 2 nulls -> 'N/A'
columns with Yes and nulls, -> boolean

concate stree 1 and street 2 with street 2 is not null

ACCLASS is the target column

"""

# Drop ACCLOC to reduce multicollinearity (corealates to LOCCOORD)
data.drop(['X', 'Y', 'INDEX_', 'ACCNUM', 'ACCLOC', 'INITDIR', 'OFFSET', 'WARDNUM', 'LATITUDE', 'LONGITUDE', 'FATAL_NO', 'INJURY', 'HOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'ObjectId'],
          axis=1, inplace=True)  # dropping irrelevent columns


# transform target col -> fatal = 1, everything else = 0

data['ACCLASS'] = np.where(data['ACCLASS'] == 'Fatal', 1, 0)

# extract features from date and time


# Convert DATE column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

data['MONTH'] = data['DATE'].dt.month
data['DAY_OF_WEEK'] = data['DATE'].dt.dayofweek  # Monday=0, Sunday=6

# Assuming rush hour is between 7-9 and 16-18 (4PM - 6PM)


def is_rush_hour(time):
    if (700 <= time <= 900) or (1600 <= time <= 1800):
        return 1
    else:
        return 0


# convert time from str to int
data['TIME'] = data['TIME'].astype(int)

data['IS_RUSH_HR'] = data['TIME'].apply(is_rush_hour)

# drop the date and time column
data.drop(['DATE', 'TIME'], axis=1, inplace=True)

data['STREET2'] = data['STREET2'].isna().map(
    {True: 0, False: 1})  # make the second street into boolean

# Identify the numerical columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Identify the categorical columns
cat_cols = data.select_dtypes(include='object').columns.tolist()


# transform cols that only has one unique value to boolean columns (Column values are either 'Yes' or null)
for col in cat_cols:
    if data[col].nunique() == 1:
        data[col].replace('Yes', 1, inplace=True)
        data[col].fillna(0, inplace=True)


# Pipeline for numerical columns
num_pipeline = Pipeline([
    # Fill missing values using mean
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # Standard scaler for numerical columns
])


# Pipeline for categorical columns
cat_pipeline = Pipeline([
    # Fill missing values by the most frequent value
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    # One hot encoder
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, cat_cols),

], remainder='passthrough')

# Transform the data
df_transformed = preprocessor.fit_transform(data)

y = data['ACCLASS']
X = data.drop('ACCLASS', axis=1)
"""
Split the data
"""
smote = SMOTE()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Apply SMOTE only on training data
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_smote))
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:03:16 2023

@author: Alex Hoang
@author: Aaron Yi-Lin Lou (301226659)
"""


# from sklearn.metrics.pairwise import _euclidean_distances
# used to balance data by oversampling using KNN

# Describe data
os.getcwd()

# define the path that locates the folder
path = r'C:/Users/lou22/Documents/SoftwareEngineeringEducation/ClassesSummer2023/COMP247SupLearn/Collision-Injury-Predictor'
os.chdir(path)
data = pd.read_csv(r"./Dataset/KSI.csv")


print("Column names:\n")
print(data.columns.values)
print("Data shape: \n")
print(data.shape)
print("Describe the data: \n")
print(data.describe(include='all'))
print("Column data types: \n")
print(data.dtypes)
print("First 5 records: \n")
for i, j in data.head(5).iterrows():
    print(i, j)
    print('------------')
print("Data information: \n")
print(data.info())

# Statistical assessment
print("Mean: \n")
print(data.mean())

print("Median: \n")
print(data.median())

print("Correlation: \n")
print(data.corr())

# Check missing values
print("Null values: \n")
print(data.isnull().sum())

# Check unique values
print("Unique values for each column: \n")
for col in data.columns:
    print(f'{col}: {data[col].nunique()}')


# Data Visualization

# To solve the problem of overlapping labels on x-axis

def wrap_labels(labels, width):
    wrapped_labels = []
    for label in labels:
        wrapped_label = '\n'.join([label[i:i+width]
                                  for i in range(0, len(label), width)])
        wrapped_labels.append(wrapped_label)
    return wrapped_labels


data.hist(figsize=(15, 20))

data['TIME'].hist()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Distribution of Time')
plt.show()

# data classes are really imbalanced
class_counts = data['INJURY'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Injury')
plt.ylabel('Count')
plt.title('Distribution of Classes')
plt.show()

cross_tab = pd.crosstab(data['VEHTYPE'], data['INJURY'])
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cross_tab, cmap='Blues', linewidths=0.5,
            linecolor="black", annot=True)
plt.xlabel('INJURY')
plt.ylabel('VEHICLE TYPE')
plt.title('Heatmap of INJURY and Vehicle Type')
plt.show()

cross_tab = pd.crosstab(data['ACCLOC'], data['ACCLASS'])
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cross_tab, cmap='Blues', linewidths=0.5,
            linecolor="black", annot=True)
plt.xlabel('FATALITY')
plt.ylabel('Accident Location')
plt.title('Heatmap of Fatality and Accident Location')
plt.show()

year_counts = data['YEAR'].value_counts()
plt.bar(year_counts.index, year_counts.values, width=0.5)
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Distribution of Accidents over Years')
plt.show()

street1_counts = data['STREET1'].value_counts().head(5)
plt.bar(wrap_labels(street1_counts.index, 9),
        street1_counts.values, width=0.5, align='center')
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
fig, ax = plt.subplots(figsize=(10, 10))
plt.bar(wrap_labels(visibility_counts.index, 11),
        visibility_counts.values, width=0.8)
plt.xlabel('Visibility')
plt.ylabel('Number of Accidents')
plt.title('Visibility Conditions over Accidents')
plt.show()

drivact_counts = data['DRIVACT'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(15, 15))
plt.bar(wrap_labels(drivact_counts.index, 16), drivact_counts.values, width=0.8)
plt.xlabel('Driver Action')
plt.ylabel('Number of Accidents')
plt.title('Top 8 Driver Action in most Accidents')
plt.show()

"""
Data modeling
"""


"""
Transformation:


Extract features: Date-> year, month, day of the week, time
time -> rush hour or not rush hour

target: fatal -> boolean
street 2 nulls -> 'N/A'
columns with Yes and nulls, -> boolean

concate stree 1 and street 2 with street 2 is not null

ACCLASS is the target column

"""
# Drop ACCLOC to reduce multicollinearity (corealates to LOCCOORD)
data.drop(['X', 'Y', 'INDEX_', 'ACCNUM', 'ACCLOC', 'INITDIR', 'OFFSET', 'WARDNUM', 'LATITUDE', 'LONGITUDE', 'FATAL_NO', 'INJURY', 'HOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'ObjectId'],
          axis=1, inplace=True)  # dropping irrelevent columns


# transform target col -> fatal = 1, everything else = 0

data['ACCLASS'] = np.where(data['ACCLASS'] == 'Fatal', 1, 0)

# extract features from date and time


# Convert DATE column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

data['MONTH'] = data['DATE'].dt.month
data['DAY_OF_WEEK'] = data['DATE'].dt.dayofweek  # Monday=0, Sunday=6

# Assuming rush hour is between 7-9 and 16-18 (4PM - 6PM)


def is_rush_hour(time):
    if (700 <= time <= 900) or (1600 <= time <= 1800):
        return 1
    else:
        return 0


print(data['TIME'].unique())
# convert time from str to int
data['TIME'] = data['TIME'].astype(int)


data['IS_RUSH_HR'] = data['TIME'].apply(is_rush_hour)

# drop the date and time column
data.drop(['DATE', 'TIME'], axis=1, inplace=True)

data['STREET2'] = data['STREET2'].isna().map(
    {True: 0, False: 1})  # make the second street into boolean

# Identify the numerical columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Identify the categorical columns
cat_cols = data.select_dtypes(include='object').columns.tolist()


# transform cols that only has one unique value to boolean columns (Column values are either 'Yes' or null)
for col in cat_cols:
    if data[col].nunique() == 1:
        data[col].replace('Yes', 1, inplace=True)
        data[col].fillna(0, inplace=True)


# Pipeline for numerical columns
num_pipeline = Pipeline([
    # Fill missing values using mean
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # Standard scaler for numerical columns
])


# Pipeline for categorical columns
cat_pipeline = Pipeline([
    # Fill missing values by the most frequent value
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    # One hot encoder
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, cat_cols),

], remainder='passthrough')

# Transform the data
df_transformed = preprocessor.fit_transform(data)

y = data['ACCLASS']
X = data.drop('ACCLASS', axis=1)
"""
Split the data
"""
smote = SMOTE()
# issues with SMOTE, it might not be compatible

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Apply SMOTE only on training data
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_smote))
