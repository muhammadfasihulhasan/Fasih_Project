import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


df = pd.read_csv('d:\DS\Programming\Fasih\Fasih_Project\weatherAUS.csv')

#checking for null values 
df.isna().sum().sort_values(ascending=False)*100/len(df)

#it was observed that the columns 'Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9pm' are having a large number of null values so removing these columns.
#along with these the 'Date' coulmn is also being dropped as it doesnot hold any significant importance in the prediction or analysis.
df.drop(["Date","Sunshine","Evaporation","Cloud3pm","Cloud9am"], axis=1, inplace=True)
#apart from these dropping the rows which have null values
df.dropna(inplace=True)

#Now the columns having string values will be dealt with as the string values cannot be fed to a machine learning model.
#The columns "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow" and "Location" are having string values.
#For now we are dealing with every column containing string values apart from "Location".
#Opting for Labelencoding technique for this purpose.

from sklearn.preprocessing import LabelEncoder
df['WindGustDir'] = LabelEncoder.fit_transform(df['WindGustDir'])
df['WindDir9am'] = LabelEncoder.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = LabelEncoder.fit_transform(df['WindDir3pm'])
df['RainToday'] = LabelEncoder.fit_transform(df['RainToday'])
df['RainTomorrow'] = LabelEncoder.fit_transform(df['RainTomorrow'])






