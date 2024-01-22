import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Rain Predictor")
st.subheader("About the Project")
st.write("This project focuses on forecasting tomorrow's rain based on the Rainfall in Australia dataset. Various machine learning models, including Logistic Regression, Decision Tree, and Random Forest have been tested. Among these models, Random Forest demonstrated exceptional performance, achieving an score of 86, surpassing the others.")

df = pd.read_csv('d:\DS\Programming\Fasih\Fasih_Project\weatherAUS.csv')

st.subheader("About the Dataset")
st.write("The first five rows of the dataset is as follows:")
st.write(df.head())

st.write("The statistical summary of the dataset is as follows:")
st.write(df.describe())

st.write(f"The dataset have {df.shape[0]} rows and {df.shape[1]} columns.")

st.subheader("Data Cleaning")
df = pd.read_csv('d:\DS\Programming\Fasih\Fasih_Project\weatherAUS.csv')

#checking for null values 
st.write("Calculating percentage of null values so that we can deal with them accordingly")
df.isna().sum().sort_values(ascending=False)*100/len(df)

#it was observed that the columns 'Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9pm' are having a large number of null values so removing these columns.
#along with these the 'Date' coulmn is also being dropped as it doesnot hold any significant importance in the prediction or analysis.
st.write("Removing columns that are having a greater percentage of null values. Along with them dropping the rows with null values. The cleaned dataset is as follows:")
df.drop(["Date","Sunshine","Evaporation","Cloud3pm","Cloud9am"], axis=1, inplace=True)
#apart from these dropping the rows which have null values
df.dropna(inplace=True)
st.write(df.head())

st.write(f"The cleaned dataset have {df.shape[0]} rows and {df.shape[1]} columns.")


#Now the columns having string values will be dealt with as the string values cannot be fed to a machine learning model.
#The columns "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow" and "Location" are having string values.
#For now we are dealing with every column containing string values apart from "Location".
#Opting for Labelencoding technique for this purpose.

st.subheader("Converting string values to numeircal values using Label Encoder.")
st.write("Below is the table representing the wind directions along with their corresponding Label Encoded Values.")

from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
df['WindGustDir'] = LabelEncoder.fit_transform(df['WindGustDir'])
df['WindDir9am'] = LabelEncoder.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = LabelEncoder.fit_transform(df['WindDir3pm'])

# Creating a table for wind directions with their label-encoded values
mapping_wind_direction = dict(zip(LabelEncoder.classes_, LabelEncoder.transform(LabelEncoder.classes_)))
st.table(pd.DataFrame(list(mapping_wind_direction.items()), columns=['Wind Direction', 'Label Encoded Numerical Value']).set_index('Wind Direction'))

df['RainToday'] = LabelEncoder.fit_transform(df['RainToday'])
df['RainTomorrow'] = LabelEncoder.fit_transform(df['RainTomorrow'])

st.write("The dataset after Label encoding is as follows.")
st.write(df.head())

st.subheader("Exploratory Data Analysis")

st.write("Now exploring/visualizing data using different types of plots") 
#Firstly a  scatter plot showing Mininmum temperature vs the Maximum temperature in which the categorical variable will be Rain Tomorrow
from pandas.plotting import scatter_matrix
plt.figure(figsize=(8,8))
sns.scatterplot(x='MaxTemp', y='MinTemp', hue='RainTomorrow', palette='inferno', data=df)
st.pyplot()
st.write("A scatter plot showing Mininmum temperature vs the Maximum temperature in which the categorical variable will be Rain Tomorrow. Through this plot we are getting a very good idea that as the minimum temperature is increasing the maximum temperature is also increasing so it is a cluster but in a linear relationship. Hence we can conclude that as the relationship is increasing linearly so this predicts that it is going to rain tomorrow.")

#Secondly a piechart for representing the data in the columnn named 'Rain Today'
import plotly.express as px
pie = px.pie(df , names= 'RainToday')
st.plotly_chart(pie)
st.write("A pie chart for representing the data in the column named 'Rain Today'.")

#Thirdly a line plot showing Humidity vs the temperature at 9 am in which the categorical variable will be Rain Tomorrow
line = sns.lineplot(x = 'Humidity9am',y = 'Temp9am', hue = 'RainTomorrow', palette = 'deep' , data = df)
st.pyplot(line.figure)
st.write("A line plot showing Humidity vs the temperature at 9 am in which the categorical variable will be Rain Tomorrow.")

#Now checking  how strongly different variables are correlated with eachother using heatmap
#The column Location has string values so dropping it for performing correlation
a = df.drop(['Location'], axis=1)
fig_heatmap, ax_heatmap = plt.subplots(figsize=(12,12))
sns.heatmap(a.corr(), annot=True, linewidths=0.1, fmt=".2f", ax = ax_heatmap)
st.pyplot(fig_heatmap)
st.write("A heatmap showing how strongly different variables are correlated with eachother. We can see that humidity have a great impact on the next day's rain.")


#Now creating a count plot showing the count of occurrences for each Location
loc_count = plt.figure(figsize=(15,15))
sns.countplot(x='Location', data= df, palette= 'Set1')
plt.xticks(rotation = 90)
plt.title('Location Count')
st.pyplot(loc_count)
st.write("A count plot showing the count of occurences for each location.")


#Now creating a count plot showing the count of occurrences of Rain to be on the next day or not
rain_tom = plt.figure(figsize=(8,8))
sns.countplot(x='RainTomorrow', data= df, palette= 'Set2')
plt.title('Rain Tomorrow Count')
st.pyplot(rain_tom)
st.write("A count plot showing the count of occurences for Rain to be on the next day or not.")


#Now using a scatterplot for the relationship between both minimum and maximum temperatures with rainfall
temp_rain = plt.figure(figsize=(12,8))
plt.scatter (df['MinTemp'], df["Rainfall"],color ="deeppink")
plt.scatter (df['MaxTemp'], df["Rainfall"],color ="mistyrose")
plt.title('Comparison of temperatures with rain')
plt.xlabel(' temp(°C)')
plt.ylabel('rain(mm)')
plt.xticks(range(-10,50,2), rotation = 30)
plt.legend(['Min temperature','Max temperature'],loc = 'upper right')
st.pyplot(temp_rain)
st.write("A scatterplot for the relationship between both minimum and maximum temperatures with rainfall.")


#Now for the relationship between Location and Rainfall
loc_rain = plt.figure(figsize=(16,12))
plt.scatter (df["Location"], df["Rainfall"],color ="deeppink")
plt.title("'Comparison of location with rain'")
plt.xlabel("location")
plt.ylabel("Rain(mm)")
plt.xticks(rotation = 90)
plt.yticks(range(0,350,20))
plt.grid()
st.pyplot(loc_rain)
st.write("A scatterplot for the relationship between Location and Rainfall.")

df['Location'] = LabelEncoder.fit_transform(df['Location'])

#Now dividing the data into dependant and independant variables
X = df.drop(['RainTomorrow'], axis=1)
Y = df['RainTomorrow']

#Now splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2)

#Now importing classification report, confusion matrix and accuracy score for metrics purpose
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

st.subheader("Prediction Models")
st.write("There are 3 models that have been used below, out of which we will select the one with the best accuracy.")

st.subheader("1. Decision Tree Classifier")
from sklearn.tree import DecisionTreeClassifier
model_1 = DecisionTreeClassifier()
model_1.fit(X_train,Y_train)
predictions = model_1.predict(X_test)
accuracy = accuracy_score(Y_test,predictions)
st.write("Accuracy Score: ", accuracy, ".")


st.subheader("2. Logistic Regression")
from sklearn.linear_model import LogisticRegression
model_2 = LogisticRegression(max_iter=1000)
model_2.fit(X_train,Y_train)
predictions = model_2.predict(X_test)
accuracy = accuracy_score(Y_test,predictions)
st.write("Accuracy Score: ", accuracy, ".")

st.subheader("3. Random Forest Classifier")
from sklearn.ensemble import RandomForestClassifier 
model_3 = RandomForestClassifier()
model_3.fit(X_train,Y_train)
predictions = model_3.predict(X_test)
accuracy = accuracy_score(Y_test,predictions)
st.write("Accuracy Score: ", accuracy, ".")

st.write("As said above, we will use the Random Forest Classifier for our rain prediction as it is the most accurate one.")
st.write("The list of the places in Australia where the rain is to be predicted along with their corresponding Label encoded numerical value is as follows.")

mapping_location = dict(zip(LabelEncoder.classes_, LabelEncoder.transform(LabelEncoder.classes_)))
st.table(pd.DataFrame(list(mapping_location.items()), columns = ['Location', 'Label Encoded Numerical Value']).set_index('Location'))

def predict(Location, minTemp, maxTemp, rainfall,
            windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
            humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm,
            rainToday):
    input_lst = [Location, minTemp, maxTemp, rainfall,
                 windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
                 humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm,
                 rainToday]
    pred = model_3.predict([input_lst])
    return pred[0]

def main():
    st.title("Weather Prediction App")

    Location = st.number_input("Location", format="%d", step=1)
    minTemp = st.number_input("Min Temperature", min_value=-10.0, max_value=50.0)
    maxTemp = st.number_input("Max Temperature", min_value=-10.0, max_value=50.0)
    rainfall = st.number_input("Rainfall", min_value=0.0, max_value=500.0)

    windGustDir = st.number_input("Wind Gust Direction", min_value=0, max_value=360)
    windGustSpeed = st.number_input("Wind Gust Speed", min_value=0.0, max_value=200.0)
    winddDir9am = st.number_input("Wind Direction 9am", min_value=0, max_value=360)
    winddDir3pm = st.number_input("Wind Direction 3pm", min_value=0, max_value=360)
    windSpeed9am = st.number_input("Wind Speed 9am", min_value=0.0, max_value=200.0)
    windSpeed3pm = st.number_input("Wind Speed 3pm", min_value=0.0, max_value=200.0)

    humidity9am = st.number_input("Humidity 9am", min_value=0.0, max_value=100.0)
    humidity3pm = st.number_input("Humidity 3pm", min_value=0.0, max_value=100.0)
    pressure9am = st.number_input("Pressure 9am", min_value=800.0, max_value=1200.0)
    pressure3pm = st.number_input("Pressure 3pm", min_value=800.0, max_value=1200.0)

    temp9am = st.number_input("Temperature 9am", min_value=-10.0, max_value=50.0)
    temp3pm = st.number_input("Temperature 3pm", min_value=-10.0, max_value=50.0)

    rainToday = st.number_input("Rain Today", min_value=0.0, max_value=1.0)

    if st.button("Predict"):
        prediction = predict(Location, minTemp, maxTemp, rainfall,
                             windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
                             humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm,
                             rainToday)

        if prediction == 0:
            st.success("It's likely to be sunny!")
        else:
            st.success("It's likely to be rainy!")

if __name__ == '__main__':
    main()



















