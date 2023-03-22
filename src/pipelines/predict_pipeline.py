import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/proprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Airline: str,
        Dep_Time,
        Arrival_Time,
        Source: str,
        Destination: str,
        Total_Stops:int,
        ):

        self.Airline = Airline

        self.Dep_Time = Dep_Time

        self.Arrival_Time = Arrival_Time

        self.Source = Source

        self.Destination = Destination
        self.Total_Stops = Total_Stops

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.Airline],
                "Dep_Time": [self.Dep_Time],
                "Arrival_Time": [self.Arrival_Time],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Total_Stops": [self.Total_Stops],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)