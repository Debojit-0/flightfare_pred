import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig
#from src.components.data_cleaning import DataCleaning


df1=pd.read_excel("/config/workspace/notebook/data/Data_Train.xlsx")


df1['Journey_day']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.day
df1['Journey_month']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.month
df1['Journey_year']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.year
df1= df1.drop(['Date_of_Journey'], axis=1)
df1['hours']=pd.to_datetime(df1['Dep_Time']).dt.hour
df1['minutes']=pd.to_datetime(df1['Dep_Time']).dt.minute
df1.drop(["Dep_Time"], axis = 1, inplace = True)
df1["Arrival_hour"] = pd.to_datetime(df1.Arrival_Time).dt.hour
df1["Arrival_min"] = pd.to_datetime(df1.Arrival_Time).dt.minute
duration = list(df1["Duration"])
#print(duration)

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]: # Check if duration contains only hour or mins
            duration[i] = duration[i].strip() + " 0m"  
        else: 
            duration[i] = "0h " + duration[i] 
    #print(duration[i])
duration_hours =[]
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
df1["duration-mins"]= duration_mins
print(df1)

