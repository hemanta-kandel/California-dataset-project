import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

# Loading Data
file = pd.read_csv('housing.csv')
data = pd.DataFrame(file)

# Main ececutable function where all displaying work will be shown
def Firstpg(average_data, common_data):
    st.title("California Household")

    st.subheader("Average data of housing.csv")
    st.write(f"The average bedroom : {average_data.get('average_bedroom')[0]:.2f}")
    st.write(f"The average number of room : {average_data.get('average_number_of_room')[0]:.2f}")
    st.write(f"The average population : {average_data.get('average_population')[0]:.2f}")
    st.write(f"The average household : {average_data.get('average_household')[0]:.2f}")
    st.write(f"The average median income : {average_data.get('average_median_income')[0]:.4f}")
    st.write(f"The average median house value : {average_data.get('average_median_house_value')[0]:.2f}")

    st.subheader('Other Important Information')
    st.write(f"Highest Longitude: {common_data.get('highest_longitude')}. Lowest Longitude: {common_data.get('lowest_longitude')}.")
    st.write(f"Highest Latitude: {common_data.get('highest_latitude')}. Lowest Latitude: {common_data.get('lowest_latitude')}.")
    st.write(f"The number of housing near sea: {common_data.get('number_near_sea')}. Further sea: {common_data.get('number_far_sea')}.")
    st.write(f"The most expensive housing come with median value {common_data.get('expensive_median_house_value')}.")
    st.write(f"The cheapest housing come with median value {common_data.get('cheapest_median_house_value')}.")

    plotting_map = data[['latitude', 'longitude']].dropna()
    st.map(plotting_map, latitude="latitude", longitude="longitude")

# Cleaning of data if there any missing values
def Cleaning():
    global data
    if data.isnull().sum().sum() >= 1:
        data.fillna(data.mean(numeric_only=True), inplace=True)
    else:
        print("No null value reported")

# Function for storing the max and min data in common_data then returning it.
def common_display(): 
    near_sea_count = int((data['ocean_proximity'] == 'NEAR BAY').sum())
    
    common_data = {
        "highest_longitude" : np.max(data['longitude']),
        "lowest_longitude" : np.min(data['longitude']),
        "highest_latitude" : np.max(data['latitude']),
        "lowest_latitude" : np.min(data['latitude']),
        "number_near_sea" : near_sea_count,
        "number_far_sea" : int(data['ocean_proximity'].isin(['INLAND', '<1H OCEAN']).sum()),
        "expensive_median_house_value" : np.max(data['median_house_value']),
        "cheapest_median_house_value" : np.min(data['median_house_value'])
    }
    return common_data

# Average data of the dataset.
def numeric_average():
    average_data = {
        "average_bedroom" : [np.mean(data['total_bedrooms'])],
        "average_number_of_room" : [np.mean(data['total_rooms'])],
        "average_population" : [np.mean(data['population'])],
        "average_household" : [np.mean(data['households'])],
        "average_median_income" : [np.mean(data['median_income'])],
        "average_median_house_value" : [np.mean(data['median_house_value'])]
    }
    return average_data

# Plotting of dataset
def plotting_difference():
    sample_data = data.head(15)
    
    total_bedroom = np.array(sample_data['total_bedrooms'])
    population = np.array(sample_data['population'])
    household = np.array(sample_data['households'])
    median_income = np.array(sample_data['median_income'])
    median_house_value = np.array(sample_data['median_house_value'])

    n_group = len(population)
    indices = np.arange(n_group)
    width = 0.15

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(indices - 2*width, total_bedroom, width, label="Total Bedroom")
    ax.bar(indices - width, population, width, label="Population")
    ax.bar(indices, household, width, label="Household")
    ax.bar(indices + width, median_income, width, label="Median income")
    ax.bar(indices + 2*width, median_house_value, width, label="Median House Value" )

    ax.set_xticks(indices)
    ax.set_xticklabels([f"prop {i}" for i in range(n_group)])
    ax.legend()

    st.pyplot(fig)

# For making a site take less time to execute by using cache
@st.cache_resource
# Model initilizing and fitting
def train_model_cached(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=4)
    model.fit(X_train, y_train)
    return model

Cleaning()
average_data = numeric_average()
common_data = common_display()

Firstpg(average_data, common_data)
plotting_difference()

X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
Y = data['median_house_value']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

st.info('Calculating model predictions, please wait...')

model = train_model_cached(xtrain, ytrain)

y_pred = model.predict(xtest)
r2_score_testing = r2_score(ytest, y_pred)
rmse = root_mean_squared_error(ytest, y_pred)

st.header('Model Training and Testing')
st.subheader('Fitting')
st.write(' - The data housing.csv have successfully fitted to model')
st.write(' - with n_estimator=100, max_depth=15, and n_jobs=-1')
st.subheader('Training')
st.write(' - Predicted data for xtest:', y_pred)
st.write(f' - R^2 score: {r2_score_testing:.4f}')
st.write(f' - RMSE (Error in $): {rmse:.2f}')


#	Website working steps
# In first i have created a main function named firstpg which is used for display of dataset and visualization.
# In second i have created a function named cleaning that replace unusual and unfilled data with meaning ful
# In third i have created a function named common display that display store the data about minimum and maximum values.
# In fourth i have created a function named numeric average for displaying of average data from the dataset
# In fifth i have created a function named plotting difference used for display of data in interacive charts
# In sixth i have created train model cached for initilizing model and fitting dataset then to use same data for making a fast response dashboard.

