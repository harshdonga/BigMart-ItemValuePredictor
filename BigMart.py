# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:12:35 2019

@author: Harsh
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


traindataset = pd.read_csv('C://Users/Harsh/Desktop/Trials/BIGMART/train.csv')
testdataset = pd.read_csv('C://Users/Harsh/Desktop/Trials/BIGMART/test.csv')
dataset = pd.concat([traindataset,testdataset],ignore_index = True)

dataset['Item_Weight'].fillna(dataset['Item_Weight'].std(), inplace = True)

dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace(('Low Fat' , 'Regular' , 'LF' , 'reg' , 'low fat'), ('Low Fat' , 'Regular','Low Fat' , 'Regular','Low Fat'))

dataset['Item_Type_Combined'] = dataset['Item_Identifier'].apply(lambda x: x[0:2])
dataset['Item_Type_Combined'] = dataset['Item_Type_Combined'].map({'DR':'Drinks' ,'FD':'Food','NC':'Non-Consumable', 'Fo':'Food','No':'Non-Consumable','Dr':'Drinks'})
dataset['Item_Type_Combined'].value_counts()

dataset.loc[dataset['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
dataset['Item_Fat_Content'].value_counts()

dataset.isnull().sum()


numeric_features = traindataset.select_dtypes(include = [np.number])

corrscore = numeric_features.corr()

corrscore['Item_Outlet_Sales'].sort_values(ascending = False)

sns.countplot(traindataset.Item_Type)
plt.xticks(rotation=90)

sns.countplot(traindataset.Outlet_Type)
plt.xticks(rotation=90)

plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(traindataset.Item_Weight, traindataset["Item_Outlet_Sales"],'.', alpha = 0.3)

from scipy.stats import mode
outlet_size_mode = dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode

def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
dataset['Outlet_Size'] = dataset[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset['Outlet'] = le.fit_transform(dataset['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])

dataset.drop(['Item_Type', 'Item_Identifier','Outlet_Identifier', 'Outlet_Establishment_Year'], axis = 1 , inplace= True)

train = dataset[:8523]
test = dataset[8523:]

trainingset = train
testingset = test

test.drop(['Item_Outlet_Sales'] , axis =1 , inplace = True)

train.to_csv('C://Users/Harsh/Desktop/Trials/BIGMART/Train_modified.csv' ,index = False)
test.to_csv('C://Users/Harsh/Desktop/Trials/BIGMART/Test_modified.csv' ,index = False)
train.columns

features = train[['Item_Fat_Content', 'Item_MRP', 'Item_Visibility',
       'Item_Weight', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
       'Item_Type_Combined', 'Outlet']].values
feature_names =['Item_Fat_Content', 'Item_MRP', 'Item_Visibility',
       'Item_Weight', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
       'Item_Type_Combined', 'Outlet']
testfeatures = test(['Item_Fat_Content', 'Item_MRP', 'Item_Visibility', 'Item_Weight',
       'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
       'Item_Type_Combined', 'Outlet'])
target = train['Item_Outlet_Sales'].values

#modelBuilding

from sklearn.linear_model import LinearRegression , Lasso
lr = LinearRegression()
lr.fit(features,target)

lr.score(features,target)   #56.36

las = Lasso(alpha=0.1)
las.fit(features,target)
las.predict(features)

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=1234,max_depth=20,min_samples_leaf=100)
dtr.fit(features, target)
dtr.score(features,target)  #61.55

from xgboost.sklearn import XGBRegressor
model = XGBRegressor(n_estimators=100, random_state=1234,max_depth=20)
model.fit(features, target)   #99.7

test['Predicted Values'] = model.predict(testfeatures)
testdataset.join(test['Predicted Values'])

testdataset.to_csv('C://Users/Harsh/Desktop/Trials/BIGMART/output.csv' ,index = False)

