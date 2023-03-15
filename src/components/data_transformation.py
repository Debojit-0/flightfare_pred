import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:


    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def clean_data(df):
        try:
            df =pd.read_csv("artifacts/data.csv")
            
            df['Journey_day']=pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.day
            df['Journey_month']=pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.month
            df['Journey_year']=pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.year
            df= df.drop(['Date_of_Journey'], axis=1)
            df['hours']=pd.to_datetime(df['Dep_Time']).dt.hour
            df['minutes']=pd.to_datetime(df['Dep_Time']).dt.minute
            df.drop(["Dep_Time"], axis = 1, inplace = True)
            df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
            df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
            duration = list(df["Duration"])

            for i in range(len(duration)):
                
                if len(duration[i].split()) != 2:
                      # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                    
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        
                        duration[i] = "0h " + duration[i]  
                        
            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
                df["duration-mins"]= duration_mins
                df["duration-hours"]= duration_hours
                df = df.drop(["Duration"],axis=1)
                df = df.drop(["Arrival_Time"],axis=1)   
            
            return df # Adds 0 hour

    def get_data_transformation(self):

        try:
            numerical_columns = ["Journey_day", "Journey_month","Journey_year","hours","minutes","Arrival_hour","Arrival_min","duration-mins","duration-hours"]
            categorical_columns = [
                "Airline",
                "Source",
                "Destination",
                "Total_Stops",
            
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            logging.info("Train test split initiated")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys) 
       
    def inititate_data_transformation(self,raw_data_path):
        try:

            df1=pd.read_csv(raw_data_path)
            cleaned_data=self.clean_data(df1)
            train_set,test_set=train_test_split(cleaned_data,test_size=0.2,random_state=42)
            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)



            #test_df=pd.resd_csv(test_path)

            logging.info("Read the raw data")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name ="Price"
            numerical_columns = ["Journey_day", "Journey_month","Journey_year","hours","minutes","Arrival_hour","Arrival_min","duration-mins","duration-hours"]
            input_feature_train_df=train_set.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_set[target_column_name]
            input_feature_test_df=test_set.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_set[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Saved preprocessing object.")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(

            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

            return (
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            
        )

        except Exception as e:
            raise CustomException(e,sys)





    

    





    








    

        
