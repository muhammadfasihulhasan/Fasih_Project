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

#Now exploring/visualizing data using different types of plots 
#Firstly a  scatter plot showing Mininmum temperature vs the Maximum temperature in which the categorical variable will be Rain Tomorrow
from pandas.plotting import scatter_matrix
#plt.figure(figsize=(8,8))
#sns.scatterplot(x='MaxTemp', y='MinTemp', hue='RainTomorrow', palette='inferno', data=df)

#Secondly a piechart for representing the data in the columnn named 'Rain Today'
import plotly.express as px
#px.pie(df , names= 'RainToday')

#Thirdly a line plot showing Humidity vs the temperature at 9 am in which the categorical variable will be Rain Tomorrow
#sns.lineplot(x = 'Humidity9am',y = 'Temp9am', hue = 'RainTomorrow', palette = 'deep' , data = df)

#Now checking  how strongly different variables are correlated with eachother using heatmap
#The column Location has string values so dropping it for performing correlation
#a = df.drop(['Location'], axis=1)
#plt.figure(figsize=(12,12))
#sns.heatmap(a.corr(), annot=True, linewidths=0.1, fmt=".2f")

#Now creating a count plot showing the count of occurrences for each Location
#plt.figure(figsize=(15,15))
#x_loc = sns.countplot(x='Location', data= df, palette= 'Set1')
#plt.xticks(rotation = 90)
#plt.title('Location Count')

#Now creating a count plot showing the count of occurrences of Rain to be on the next day or not
#x_tom = sns.countplot(x='RainTomorrow', data= df, palette= 'Set2')
#plt.title('Rain Tomorrow Count')

#Now using a scatterplot for the relationship between both minimum and maximum temperatures with rainfall
#plt.figure(figsize=(12,8))
#plt.scatter (df['MinTemp'], df["Rainfall"],color ="deeppink")
#plt.scatter (df['MaxTemp'], df["Rainfall"],color ="mistyrose")
#plt.title('Comparison of temperatures with rain')
#plt.xlabel(' temp(°C)')
#plt.ylabel('rain(mm)')
#plt.xticks(range(-10,50,2), rotation = 30)
#plt.legend(['Min temperature','Max temperature'],loc = 'upper right')

#Now for the relationship between Location and Rainfall
#plt.figure(figsize=(16,12))
#plt.scatter (df["Location"], df["Rainfall"],color ="deeppink")
#plt.title("'Comparison location with rain'")
#plt.xlabel("location")
#plt.ylabel("Rain(mm)")
#plt.xticks(rotation = 90)
#plt.yticks(range(0,350,20))
#plt.grid()

#Now splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2)

#Now importing classification report, confusion matrix and accuracy score for metrics purpose
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

#Now the first model we will be using is Decision Tree
#model_1 = DecisionTreeClassifier()
#model_1.fit(X_train,Y_train)
#predictions = model_1.predict(X_test)
#print(confusion_matrix(Y_test,predictions))
#print(classification_report(Y_test,predictions))
#print(accuracy_score(Y_test,predictions))

#Now the second model we will be using is Logistic Regression
#model = LogisticRegression(max_iter=1000)
#model.fit(X_train,Y_train)
#predictions = model.predict(X_test)
#print(confusion_matrix(Y_test,predictions))
#print(classification_report(Y_test,predictions))
#print(accuracy_score(Y_test,predictions))














