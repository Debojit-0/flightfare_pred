import pandas as pd
from src.exception import CustomException

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

df=pd.read_excel("/config/workspace/notebook/data/Data_Train.xlsx")
df1=df.copy()
print(df1)

df1['Journey_day']=pd.to_datetime(df1['Date_of_Journey'],format="%d/%m/%Y").dt.day
print(df1)